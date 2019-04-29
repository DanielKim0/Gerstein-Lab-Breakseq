from keras.optimizers import SGD
from sklearn.metrics import normalized_mutual_info_score
from MLModel import MLModel
import os
from dec_algorithm import DEC
import sys
import numpy as np

class DECModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_dec.txt", 0.3, True)
        self.initializer = 'glorot_uniform'
        self.pretrain_optimizer = 'adam'
        self.batch_size = 128
        self.maxiter = 2e4
        self.tol = 0.001
        self.save_dir = "results_dec"
        self.update_interval = 2
        self.pretrain_epochs = 1
        if not os.path.exists("results_dec"):
            os.makedirs("results_dec")

    def model_build(self, clusters):
        dec = DEC(dims=[self.X_train.shape[-1], 50, 50, 200, clusters], n_clusters=clusters, init=self.initializer)
        dec.pretrain(x=self.X_train, y=self.y_train, optimizer=self.pretrain_optimizer, epochs=self.pretrain_epochs,
                     batch_size=self.batch_size, save_dir=self.save_dir)
        dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')
        return dec

    def model_run(self, model):
        model.fit(self.X_train, y=self.y_train, tol=self.tol, maxiter=self.maxiter, batch_size=self.batch_size,
                  update_interval=self.update_interval, save_dir=self.save_dir)
        pred_val = model.predict(self.X_test)

        self.results.write("Model Results\n")
        self.results.write("NMI Score: " + str(normalized_mutual_info_score(self.y_test, pred_val)) + "\n")
        return model

    def model_probs(self, model=None):
        if not model:
            model = self.model_build(3)
            model.fit(self.X_train, y=self.y_train, tol=self.tol, maxiter=self.maxiter, batch_size=self.batch_size,
                      update_interval=self.update_interval, save_dir=self.save_dir)
        predictions = model.predict_proba(self.X_test)
        return predictions

    def kfold_run(self, clusters):
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

if __name__ == "__main__":
    dec = DECModel(sys.argv[1])
    dec_model = dec.model_build(3)
    dec_model_fit = dec.model_run(dec_model)
    dec.kfold_run(3)
    probs = dec.model_probs(dec_model_fit)
    dec.roc(probs, "DEC ROC Graph", "dec_roc.png")