# Bio Research Steps Taken

## Breakseq Generation

The dataset used was the reference genome hg38 from the Genome Reference Consortium, located here: http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz

We used breakseq to first generate a breakseq library from the hg38 .fa genome , whose chromosomes are located here, http://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/, using reference B38 .gff files located here http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/integrated_sv_map/supporting/GRCh38_positions/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.GRCh38.vcf.gz. Then, we ran breakseq itself with these <ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data>/ reference BAM files and these <ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz> reference fasta files. We used the resulting output files for the next step, converting vcf to gff files when needed.

## Data Accumulation

Breakseq generated its own statistics of the data, including flexibility, duplex stability, and GC content. Here, the breakseq deletions and insertions were elongated into left junctions (500 bases past the beginning and 50 past the end) and right junctions (50 bases past the beginning and 500 past the end), with one of each junction per sequence. 

Two rounds of feature matrix generation for shape determination using various bigwig files were ran on both left and right sequences. Sequence complexity was then computed using this code <https://github.com/caballero/SeqComplex>. TAD boundary data and fragile site data were used to compute intersections for TAD boundaries. Additionally, recombination hotspot motif occurrences were calculated for the following sequence: "CCNCCNTNNCCNC" (where N is any nucleotide). After this, the data was coalesced and both left and right sequence data combined into one large data file with many various data points.

## Machine Learning

This data file was used to calculate various characteristics about the breakseq-given classifications: NHR, NAHR, and NAHR_EXT. From here, various machine learning models were created to test the accuracy of various models for classifying new sequences to these labels. These models include:

- A random forest classifier
- A K-Means unsupervised learner
- A SVC, which was used to calculate various curves, including ROC and AUC
- CNNs with varying amounts of layers
- A DEC, or Deep Embedding for Clustering algorithm. (More info can be found here: https://arxiv.org/abs/1511.06335)
- An autoencoder-based model

Models primarily used scikit-learn and/or keras as their base. Additionally, for each model, ROC graphs were created to give insight onto veracity of models, and hyperparameter-tuning methods were added to lead to further accuracy.