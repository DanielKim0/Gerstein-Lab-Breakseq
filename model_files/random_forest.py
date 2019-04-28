from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
from MLModel import MLModel
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


class RandomForestModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_random_forest.txt", 0.3)

    def model_build(self, estimators):
        """
        Returns the requisite Random Forest object with the inputted number of estimators.
        """
        return RandomForestClassifier(n_estimators=estimators, n_jobs=-1)

    def model_run(self, model, estimators):
        """
        Runs the model using the above model_build function and fits/predicts the data using it.
        """
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
        """
        Inputs a classifer of the type used above in model_run and returns the probabilities that each item in the dataset belongs to each class.
        """
        if not classifier:
            classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            classifier.fit(self.X_train, self.y_train)
        predictions = classifier.predict_proba(self.X)
        return predictions

    def kfold_run(self, estimators):
        """
        Runs kfold cross-validation using the generated model.
        """
        model = self.model_build(estimators)
        super().kfold_run(model)

    def random_param_tune(self):
        """
        Creates, fits, and predicts a model multiple times with random combinations of hyperparameters, given below,
        in an attempt to find the best set of hyperparameters from a wide range of possibilities.
        """
        random_grid = {'bootstrap': [True, False],
         'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
         'max_features': ['auto', 'sqrt'],
         'min_samples_leaf': [1, 2, 4],
         'min_samples_split': [2, 5, 10],
         'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=250, cv=3, verbose=2, n_jobs=-1)
        rf_random.fit(self.X_train, self.y_train)
        self.results.write(str(rf_random.best_params_) + "\n")

    def param_tune(self):
        """
        Creates, fits, and predicts a model multiple times with every combination of hyperparameters, given below,
        in an attempt to fine-tune the model using more precise possibilities than the random tuning above.
        """
        grid = {'bootstrap': [True, False],
         'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
         'max_features': ['auto', 'sqrt'],
         'min_samples_leaf': [1, 2, 4],
         'min_samples_split': [2, 5, 10],
         'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

        rf = RandomForestClassifier()
        rf_grid = GridSearchCV(estimator=rf, param_distributions=grid, verbose=2, n_jobs=-1)
        rf_grid.fit(self.X_train, self.y_train)
        self.results.write(str(rf_grid.best_params_) + "\n")

if __name__ == "__main__":
    rf = RandomForestModel(sys.argv[1])
    rf_model = rf.model_build(100)
    rf_model_fit = rf.model_run(rf_model, 100)
    rf.kfold_run(100)

    probs = rf.model_probs(classifier=rf_model_fit)
    rf.roc(probs, "Random Forest ROC Graph", "random_forest_roc.png", change_probs=True)

    rf.random_param_tune()
