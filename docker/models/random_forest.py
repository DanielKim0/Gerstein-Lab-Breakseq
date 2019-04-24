from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
from MLModel import MLModel


class RandomForestModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_random_forest.txt", 0.3)

    def model_build(self, estimators):
        return RandomForestClassifier(n_estimators=estimators)

    def model_run(self, model, estimators):
        model.fit(self.X_train, self.y_train)
        y_score = model.predict(self.X_test)
        accu_train = np.sum(model.predict(self.X_train) == self.y_train) / self.y_train.size
        accu_test = np.sum(y_score == self.y_test) / self.y_test.size

        self.results.write("Model Results\n")
        self.results.write("Number of Estimators: " + str(estimators) + "\n")
        self.results.write("Accuracy on Train: " + str(accu_train) + "\n")
        self.results.write("Accuracy on Test: " + str(accu_test) + "\n")
        return model

    def model_probs(self, classifier=None):
        if not classifier:
            classifier = RandomForestClassifier(n_estimators=100)
            classifier.fit(self.X_train, self.y_train)
        predictions = classifier.predict_proba(self.X_test)
        return predictions

    def kfold_run(self, estimators):
        model = self.model_build(estimators)
        super().kfold_run(model)

if __name__ == "__main__":
    rf = RandomForestModel(sys.argv[1])
    rf_model = rf.model_build(100)
    rf_model_fit = rf.model_run(rf_model, 100)
    rf.kfold_run(100)
    probs = rf.model_probs(classifier=rf_model_fit)
    rf.roc(probs, "Random Forest ROC Graph", "random_forest_roc.png")

