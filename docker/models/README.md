# Models Docker Image

This is a docker image that runs the Python-based machine learning scripts present in its grandparent folder. Find out more information about the scripts there.

The docker image accepts one environment variable: DATA, the path to the CSV file that you want to use. The requirements for this CSV are the same as the requirements for the general CSV in the grandparent folder: the leftmost column is a dummy column (with no data) and the rightmost column is the column containing the labels to classify.

Run the script using this format: *docker run -e DATA=path_to_data*