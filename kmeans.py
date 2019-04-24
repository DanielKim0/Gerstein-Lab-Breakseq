from sklearn.cluster import KMeans
import sys
from MLModel import MLModel
from Plots import roc, prc
from keras.utils import to_categorical
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

class KMeansModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_kmeans.txt", 0, True)

    def model_build(self, clusters):
        """
        Returns the requisite KMeans object with the inputted number of clusters.
        """
        return KMeans(n_clusters=clusters)

    def model_run(self, model):
        """
        Runs the KMeans model using the above model_build function and fits the data to clusters using it.
        """
        clusters = model.fit_predict(self.X)
        centroids = model.cluster_centers_

        self.results.write("Model Results\n")
        self.results.write("Clusters: " + str(clusters) + "\n")
        self.results.write("Centroids: " + str(centroids) + "\n")
        return clusters

    def kfold_run(self, clusters):
        """
        Runs kfold cross-validation using the generated KMeans model.
        """
        kmeans_model = kmeans.model_build(clusters)
        super().kfold_run(kmeans_model)

    def create_roc(self, y_score):
        """
        Creates a ROC graph using the predicted results of the KMeans model.
        """
        roc(to_categorical(self.y), to_categorical(y_score))

    def random_param_tune(self, clusters):
        """
        Creates, fits, and predicts a model multiple times with random combinations of hyperparameters, given below,
        in an attempt to find the best set of hyperparameters from a wide range of possibilities.
        """
        random_grid = {'init': ["k-means++", "random"],
         'n_init': [4, 7, 10, 13, 16],
         'max_iter': [100, 200, 300, 400, 500],
         'tol': [10**-2, 10**-3, 10**-4, 10**-5, 10**-6],}

        km = KMeans(n_clusters=clusters)
        km_random = RandomizedSearchCV(estimator=km, param_distributions=random_grid, n_iter=100, cv=3, verbose=2)
        km_random.fit(self.X, self.y)
        self.results.write(str(km_random.best_params_) + "\n")

    def param_tune(self, clusters):
        """
        Creates, fits, and predicts a model multiple times with every combination of hyperparameters, given below,
        in an attempt to fine-tune the model using more precise possibilities than the random tuning above.
        """
        grid = {'init': ["k-means++", "random"],
         'n_init': [4, 7, 10, 13, 16],
         'max_iter': [100, 200, 300, 400, 500],
         'tol': [10**-2, 10**-3, 10**-4, 10**-5, 10**-6],}

        km = KMeans(n_clusters=clusters)
        km_grid = GridSearchCV(estimator=km, param_distributions=grid, verbose=2)
        km_grid.fit(self.X, self.y)
        self.results.write(str(km_grid.best_params_) + "\n")

if __name__ == "__main__":
    kmeans = KMeansModel(sys.argv[1])
    kmeans_model = kmeans.model_build(3)
    y_predicted = kmeans.model_run(kmeans_model)
    kmeans.kfold_run(3)

    kmeans.create_roc(y_predicted)

    kmeans.random_param_tune(3)
