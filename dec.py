from keras.optimizers import SGD
from sklearn.metrics import normalized_mutual_info_score
from MLModel import MLModel
import os, sys
from dec_algorithm import DEC
import numpy as np
from Plots import roc
from keras.utils import to_categorical
from itertools import product
from random import sample

class DECModel(MLModel):
    def __init__(self, data_file):
        """
        Creates the parent object and sets various hyperparemeters to use later.
        """
        super().__init__(data_file, "results_dec.txt", 0.3, True)
        self.initializer = 'glorot_uniform'
        self.pretrain_optimizer = 'adam'
        self.batch_size = 128
        self.maxiter = 2e4
        self.tol = 0.001
        self.save_dir = "results_dec"
        self.update_interval = 50
        self.pretrain_epochs = 30
        if not os.path.exists("results_dec"):
            os.makedirs("results_dec")

    def model_build(self, clusters, init=None, optimizer=None, epochs=None, batch_size=None):
        """
        Builds, pretrains, and compiles a DEC model using custom or preset hyperparameters.
        """
        params = [init, optimizer, epochs, batch_size]
        defaults = [self.initializer, self.pretrain_optimizer, self.pretrain_epochs, self.batch_size]
        for i in range(4):
            if params[i] is None:
                params[i] = defaults[i]

        dec = DEC(dims=[self.X_train.shape[-1], 50, 50, 200, clusters], n_clusters=clusters, init=params[0])
        dec.pretrain(x=self.X_train, y=self.y_train, optimizer=params[1], epochs=params[2],
                     batch_size=params[3], save_dir=self.save_dir)
        dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')
        return dec

    def model_run(self, model):
        """
        Runs the DEC model using the above model_build function and fits/predicts the data using it.
        """
        model.fit(self.X_train, y=self.y_train, tol=self.tol, maxiter=self.maxiter, batch_size=self.batch_size,
                  update_interval=self.update_interval, save_dir=self.save_dir)
        predicted_vals = model.predict(self.X_test)

        self.results.write("Model Results\n")
        self.results.write("NMI Score: " + str(normalized_mutual_info_score(self.y_test, predicted_vals)) + "\n")
        return model

    def kfold_run(self, clusters):
        """
        Runs kfold cross-validation using a generated DEC model and the number of clusters to use.
        """
        svscores = []
        for train, test in self.kfold.split(self.X, self.y):
            model = self.model_build(clusters)
            model.fit(self.X[train], self.y[train], tol=self.tol, maxiter=self.maxiter, batch_size=self.batch_size,
                      update_interval=self.update_interval, save_dir=self.save_dir)
            predicted = model.predict(self.X[test])
            scores = len([predicted[x] == self.y[test][x] for x in predicted]) / len(predicted)
            svscores.append(scores * 100)
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
        best_nmi = 0

        all_comb = list(product(['glorot_uniform', 'normal', 'uniform'], # initializer
                  ['rmsprop', 'adam'], # optimizer
                  [10, 20, 30, 40, 50], # epochs
                  [32, 64, 128, 256, 512], # batch size
                  [0.1, 0.01, 0.001, 0.0001, 0.00001], # tol
                  [1e4, 2e4, 3e4], # maxiter
                  [10, 30, 50, 70, 90])) # update interval
        if len(all_comb) > 250:
            all_comb = sample(all_comb, 250)

        for comb in all_comb:
            model = self.model_build(3, init=comb[0], optimizer=comb[1], epochs=comb[2], batch_size=comb[3])
            model.fit(self.X_train, y=self.y_train, tol=comb[4], maxiter=comb[5], batch_size=comb[3],
                      update_interval=comb[6], save_dir=self.save_dir)
            predicted_vals = model.predict(self.X_test)
            nmi = normalized_mutual_info_score(self.y_test, predicted_vals)
            if nmi > best_nmi:
                best_params = comb
                best_nmi = nmi
        self.results.write("Random Param Tune Results\n")
        self.results.write(str(best_params) + "\n")
        self.results.write(str(best_nmi) + "\n")

    def param_tune(self):
        """
        Creates, fits, and predicts a model multiple times with every combination of hyperparameters, given below,
        in an attempt to fine-tune the model using more precise possibilities than the random tuning above.
        """
        best_params = ()
        best_nmi = 0

        for comb in list(product(['glorot_uniform', 'normal', 'uniform'], # initializer
                  ['rmsprop', 'adam'], # optimizer
                  [10, 20, 30, 40, 50], # epochs
                  [32, 64, 128, 256, 512], # batch size
                  [0.1, 0.01, 0.001, 0.0001, 0.00001], # tol
                  [1e4, 2e4, 3e4], # maxiter
                  [10, 30, 50, 70, 90])): # update interval
            model = self.model_build(3, init=comb[0], optimizer=comb[1], epochs=comb[2], batch_size=comb[3])
            model.fit(self.X_train, y=self.y_train, tol=comb[4], maxiter=comb[5], batch_size=comb[3],
                      update_interval=comb[6], save_dir=self.save_dir)
            predicted_vals = model.predict(self.X_test)
            nmi = normalized_mutual_info_score(self.y_test, predicted_vals)
            if nmi > best_nmi:
                best_params = comb
                best_nmi = nmi
        self.results.write("Param Tune Results\n")
        self.results.write(str(best_params) + "\n")
        self.results.write(str(best_nmi) + "\n")

if __name__ == "__main__":
    dec = DECModel(sys.argv[1])
    dec_model = dec.model_build(3)
    dec_model_fit = dec.model_run(dec_model)
    dec.kfold_run(3)

    roc(to_categorical(dec.y), to_categorical(dec_model.predict(dec.X)), dec.results, "dec_roc.png")

    dec.param_tune()
