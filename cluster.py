import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import sys

def prepare_data(file_name):
    dataset = pd.read_csv(file_name)
    dataset = dataset[dataset.classifs != "\"UNSURE\""]
    dataset.dropna(axis="columns", inplace=True)
    dataset["classifs"] = dataset["classifs"].astype(str)
    X = dataset.iloc[:, 1:-1]
    y = dataset.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    kmeans = KMeans(n_clusters=3)
    return X, y, kmeans

def kfold(X, y, kmeans):
    kfold = KFold(n_splits=10)
    results = cross_val_score(kmeans, X, y, cv=kfold, scoring="accuracy")
    result = results.mean()
    std = results.std()
    results = open("kfold_cluster.txt", "w")
    results.write("mean:" + str(result) + "\n")
    results.write("std:" + str(std) + "\n")

def kmeans_results(X, kmeans):
    clusters = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    results = open("results_cluster.txt", "w")
    results.write(str(clusters) + "\n")
    results.write(str(centroids) + "\n")

def main(file_name):
    X, y, kmeans = prepare_data(file_name)
    kfold(X, y, kmeans)

if __name__ == "__main__":
    main(sys.argv[1])
