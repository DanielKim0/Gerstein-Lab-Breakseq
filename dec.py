from keras.optimizers import SGD
from sklearn.metrics import normalized_mutual_info_score
from MLModel import MLModel
import os
from dec_algorithm import DEC
import sys

class DECModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_dec.txt", 0.3)
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

    def kfold_run(self, clusters):
        model = self.model_build(clusters)
        super().kfold_run(model)

if __name__ == "__main__":
    dec = DECModel(sys.argv[1])
    dec_model = dec.model_build(3)
    dec.model_run(dec, dec_model)
    dec.kfold_run(3)
