# README
This repository is a collection of scripts that I used to assist in a machine learning biocomputational project for Gerstein Labs.

## Running Scripts

The scripts are written in and tested for Python 3. Run a script using *python name_of_model.py data.csv*, where *name_of_model.py* is the path to the Python script containing the model that you want to run (Ex. random_forest.py), and *data.csv* is the path to the CSV file containing the data.

The CSV file is assumed to fulfill the following criteria: the leftmost column is a dummy column (with no data) and the rightmost column is the column containing the labels to classify.s

You can change what parts of the model that you want to run by changing the lines under the if statement located at the bottom of each of the models, such as if you do not want to generate an ROC graph.

## Project Summary
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

## Project Specifics

### Breakseq Generation

The dataset used was the reference genome hg38 from the Genome Reference Consortium, located here: http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz

We used breakseq to first generate a breakseq library from the hg38 .fa genome , whose chromosomes are located here, http://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/, using reference B38 .gff files located here http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/integrated_sv_map/supporting/GRCh38_positions/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.gz. Then, we ran breakseq itself with these <ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data>/ reference BAM files and these <ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz> reference fasta files. We used the resulting output files for the next step, converting vcf to gff files when needed.

### Data Accumulation

Breakseq generated its own statistics of the data, including flexibility, duplex stability, and GC content. Here, the breakseq deletions and insertions were elongated into left junctions (500 bases past the beginning and 50 past the end) and right junctions (50 bases past the beginning and 500 past the end), with one of each junction per sequence. 

Two rounds of feature matrix generation for shape determination using various bigwig files were ran on both left and right sequences. Sequence complexity was then computed using this code <https://github.com/caballero/SeqComplex>. TAD boundary data and fragile site data were used to compute intersections for TAD boundaries. Additionally, recombination hotspot motif occurrences were calculated for the following sequence: "CCNCCNTNNCCNC" (where N is any nucleotide); to do this, nucleotide strings were extracted using samtools. After this, the data was coalesced and both left and right sequence data combined into one large data file with many various data points.

### Machine Learning

This data file was used to calculate various characteristics about the breakseq-given classifications: NHR, NAHR, and NAHR_EXT. From here, various machine learning models were created to test the accuracy of various models for classifying new sequences to these labels. These models include:

- A random forest classifier
- A K-Means unsupervised learner
- A SVC, which was used to calculate various curves, including ROC and AUC
- CNNs with varying amounts of layers
- A DEC, or Deep Embedding for Clustering algorithm. (More info can be found here: https://arxiv.org/abs/1511.06335)
- An autoencoder-based model

Models primarily used scikit-learn and/or keras as their base. Additionally, for each model, ROC graphs were created to give insight onto veracity of models, and hyperparameter-tuning methods were added to lead to further accuracy.