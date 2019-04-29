from sklearn.cluster import KMeans
import sys
from MLModel import MLModel
from Plots import roc, prc

class KMeansModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_kmeans.txt", 0, True)

    def model_build(self, clusters):
        return KMeans(n_clusters=clusters)

    def model_run(self, model):
        clusters = model.fit_predict(self.X)
        centroids = model.cluster_centers_

        self.results.write("Model Results\n")
        self.results.write("Clusters: " + str(clusters) + "\n")
        self.results.write("Centroids: " + str(centroids) + "\n")

    def kfold_run(self, clusters):
        kmeans_model = kmeans.model_build(3)
        super().kfold_run(kmeans_model)

    def create_roc(self, y_score):
        roc(self.results_convert(self.y_test), y_score)

if __name__ == "__main__":
    kmeans = KMeansModel(sys.argv[1])
    kmeans_model = kmeans.model_build(3)
    y_predicted = kmeans.model_run(kmeans_model)
    kmeans.kfold_run(3)
    kmeans.create_roc(y_predicted)
