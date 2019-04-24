# Use an official Python runtime as a parent image
FROM python:2.7

# Set the working directory to /app
WORKDIR /app

# Add necessary folders and files
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install https://github.com/bioinform/breakseq2/archive/2.2.tar.gz

# Run app.py when the container launches
CMD ["python", "breakseq/breakseq2-2.2/scripts/run_breakseq2.py", "--reference", "data/hs37d5.fa", "--bams", "data/HG00096.wgs.ILLUMINA.bwa.GBR.high_cov_pcr_free.20140203.bam", "--work", "recent_work", "--bwa", "breakseq/bwa-0.7.12/bwa", "--samtools", "breakseq/samtools-0.1.19/samtools", "--bplib", "results_gen_bplib"]
CMD ["python", "breakseq/breakseq2-2.2/scripts/breakseq2_gen_bplib.py", "--reference", "~/data/hg38.fa", "--bplib_gff", "data/B38.callset.public.gff", "--output", "~/B38.callset.public.fasta"]
