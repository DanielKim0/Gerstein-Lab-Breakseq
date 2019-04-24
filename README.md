# README
This repository is a collection of scripts that I used to assist in a machine learning biocomputational project for
Gerstein Labs.

## Data Collection
The data used for this project originated from the reference genome hg38 from the Genome Reference Consortium. We generated a feature matrix with data about the genome by extracting characteristics from the Genome; first, we classified all of the junctions by using the Breakseq program, and kept deletions and insertions. Our final classifications were NHR, NAHR, and NAHR_EXT. Then, we got a left junction (500 bases past the beginning and 50 past the end) and a right junction (50 bases past the beginning and 500 past the end). We collected data on those junctions, which included:
* The kinds of shapes of DNA around the breakpoint junction 
* Sequence complexity of the junctions
* What fraction of the breakpoing junction overlaps with 
* How many times a certain recombination hotspot is present in the a junction
* The fraction of overlap between TAD boundaries for each SV
* Statistics, such as GC content, helix flexibility, and stability
* Other statistics about conservation score, WGBS, and histone marks

From here, I combined all of this data into a large feature matrix and used that matrix as input for a variety of machine learning models, including:
* A random forest classifier
* A K-Means unsupervised learner
* A SVC, which was used to calculate various curves, including ROC and AUC
* CNNs with varying amounts of layers
* A DEC, or Deep Embedding for Clustering algorithm. (More info can be found here: https://arxiv.org/abs/1511.06335)
* An autoencoder-based model

Additional steps were taken with some of the models.
* Stratified K-Folding was used to generate separate models for each of the algorithms, which were then measured.
* Hyperparameter tuning was used for the random forest, cnn, and autoencoder models.
* ROC graphs were generated to give more infomation about the models.