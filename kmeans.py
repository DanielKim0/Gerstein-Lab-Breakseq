from sklearn.cluster import KMeans
import sys
from MLModel import MLModel

class KMeansModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_kmeans.txt", 0)

    def model_build(self, clusters):
        return KMeans(n_clusters=clusters)

    def model_run(self, model):
        clusters = model.fit_predict(self.X)
        centroids = model.cluster_centers_

        self.results.write("Model Results\n")
        self.results.write("Clusters: " + str(clusters) + "\n")
        self.results.write("Centroids: " + str(centroids) + "\n")

    def kfold_run(self, clusters):
        model = self.model_build(clusters)
        super().kfold_run(model)

if __name__ == "__main__":
    kmeans = KMeansModel(sys.argv[1])
    kmeans_model = kmeans.model_build(3)
    kmeans.kfold_run(kmeans_model)
