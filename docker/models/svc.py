import numpy as np
import sys
from MLModel import MLModel
from Plots import roc, prc

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

class SVCModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_svc.txt", 0.5)

    def model_build(self):
        return OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))

    def model_run(self, model):
        y_score = model.fit(self.X_train, self.y_train).decision_function(self.X_test)
        accu_train = np.sum(model.predict(self.X_train) == self.y_train) / self.y_train.size
        accu_test = np.sum(y_score == self.y_test) / self.y_test.size

        self.results.write("Model Results\n")
        self.results.write("Accuracy on Train: " + str(accu_train) + "\n")
        self.results.write("Accuracy on Test: " + str(accu_test) + "\n")

        return model, y_score

    def model_probs(self, model):
        if not model:
            model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
            model.fit(self.X_train, self.y_train).decision_function(self.X_test)
        predictions = model.predict_proba(self.X_test)
        return predictions

    def kfold_run(self):
        model = self.model_build()
        super().kfold_run(model)

if __name__ == "__main__":
    svc = SVCModel(sys.argv[1])
    svc_model = svc.model_build()
    model, y_score = svc.model_run(svc_model)
    svc.kfold_run()
    probs = svc.model_probs(model=model)
    svc.roc(probs, "SVC ROC Graph", "svc_roc.png")
