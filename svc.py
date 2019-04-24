import numpy as np
import sys
from MLModel import MLModel
from Plots import roc, prc

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

class OvRC(OneVsRestClassifier):
    """
    A wrapper class that simply returns a OneVsRestClassifier with additional special parameters for
    the inner SVC object used for the classification.
    """
    def __init__(self, *args):
        super().__init__(svm.SVC(*args))


class SVCModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_svc.txt", 0.5)

    def model_build(self):
        """
        Builds the SVC model through OneVsRestClassifier used to classify the dataset, and then returns it.
        """
        return OneVsRestClassifier(svm.SVC(kernel='linear', probability=True), n_jobs=-1)

    def model_run(self, model):
        """
        Runs the SVC model using the above model_build function and fits/predicts the data using it.
        """
        model.fit(self.X_train, self.y_train).decision_function(self.X_test)
        accu_train = np.sum(model.predict(self.X_train) == self.y_train) / self.y_train.size
        accu_test = np.sum(model.predict(self.X_test) == self.y_test) / self.y_test.size

        self.results.write("Model Results\n")
        self.results.write("Accuracy on Train: " + str(accu_train) + "\n")
        self.results.write("Accuracy on Test: " + str(accu_test) + "\n")

        return model, model.predict(self.X_test)

    def model_probs(self, model):
        """
        Inputs a classifer of the type used above in model_run and returns the probabilities that each item in the dataset belongs to each class.
        """
        if not model:
            model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True), n_jobs=-1)
            model.fit(self.X_train, self.y_train).decision_function(self.X_test)
        predictions = model.predict_proba(self.X)
        return predictions

    def kfold_run(self):
        """
        Runs kfold cross-validation using the generated SVC model.
        """
        model = self.model_build()
        super().kfold_run(model)

    def random_param_tune(self):
        """
        Creates, fits, and predicts a model multiple times with random combinations of hyperparameters, given below,
        in an attempt to find the best set of hyperparameters from a wide range of possibilities.
        """
        random_grid = {'kernels': ["linear", "rbf", "poly"],
         'gamma': [0.1, 1, 10, 100],
         'C': [0.1, 1, 10, 100, 1000],
         'degree': [0, 1, 2, 3, 4, 5, 6]}

        svc = OvRC()
        svc_random = RandomizedSearchCV(estimator=svc, param_distributions=random_grid, n_iter=50, cv=3, verbose=2, n_jobs=-1)
        svc_random.fit(self.X_train, self.y_train)
        self.results.write(str(svc_random.best_params_) + "\n")

    def param_tune(self):
        """
        Creates, fits, and predicts a model multiple times with every combination of hyperparameters, given below,
        in an attempt to fine-tune the model using more precise possibilities than the random tuning above.
        """
        grid = {'kernels': ["linear", "rbf", "poly"],
         'gamma': [0.1, 1, 10, 100],
         'C': [0.1, 1, 10, 100, 1000],
         'degree': [0, 1, 2, 3, 4, 5, 6]}

        svc = OvRC()
        svc_grid = GridSearchCV(estimator=svc, param_distributions=grid, verbose=2, n_jobs=-1)
        svc_grid.fit(self.X_train, self.y_train)
        self.results.write(str(svc_grid.best_params_) + "\n")

if __name__ == "__main__":
    svc = SVCModel(sys.argv[1])
    svc_model = svc.model_build()
    model, y_score = svc.model_run(svc_model)
    svc.kfold_run()

    probs = svc.model_probs(model=model)
    svc.roc(probs, "SVC ROC Graph", svc.results, "svc_roc.png")

    svc.random_param_tune()
