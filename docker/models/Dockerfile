# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory to /app
WORKDIR /app

# Add necessary folders and files
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run when the container launches
CMD ["python", "autoencoder.py", "data.csv"]
CMD ["python", "cnn.py", "data.csv"]
CMD ["python", "dec.py", "data.csv"]
CMD ["python", "kmeans.py", "data.csv"]
CMD ["python", "random_forest.py", "data.csv"]
CMD ["python", "svc.py", "data.csv"]
