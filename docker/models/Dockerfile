# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory to /app
WORKDIR /app

# Add necessary folders and files
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run when the container launches
CMD ["sh", "-c", "python scripts/autoencoder.py ${DATA}"]
CMD ["sh", "-c", "python scripts/cnn.py ${DATA}"]
CMD ["sh", "-c", "python scripts/dec.py ${DATA}"]
CMD ["sh", "-c", "python scripts/kmeans.py ${DATA}"]
CMD ["sh", "-c", "python scripts/random_forest.py ${DATA}"]
CMD ["sh", "-c", "python scripts/svc.py ${DATA}"]
