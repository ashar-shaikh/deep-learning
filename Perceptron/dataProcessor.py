import os
import csv
import random

class DataProcessor:
    def __init__(self, url, filename):
        self.url = url
        self.filename = filename
        self.label_map = {}

    def download_data(self):
        import urllib.request
        urllib.request.urlretrieve(self.url, self.filename)
        print(f"Data downloaded and saved as {self.filename}")

    def process_data(self, training_ratio=0.6, validation_ratio=0.2, hasID=False):
        unique_labels = set()
        data = []
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and '?' not in row:  # This also filters out rows with missing values
                    unique_labels.add(row[-1])
                    data.append(row)    

        # Ensure that there are only two unique labels for the perceptron
        if len(unique_labels) != 2:
            raise ValueError("Perceptron can handle binary classification. Found multiple labels.")

        # Create an automated map from labels to integers
        label_map = {label: idx for idx, label in enumerate(unique_labels)}

        for idx, sample in enumerate(data):
            if sample:  # Ensure that the sample is not empty
                # Extract the sample code number and the rest of the values
                if hasID:
                    #sample_code_number = sample[0]
                    features_and_label = sample[1:]
                else:
                    features_and_label = sample
                converted_sample = [int(value) for value in features_and_label[:-1]]  # Convert all except the last one
                converted_sample.append(label_map[features_and_label[-1]])  # Add the mapped label
                data[idx] = converted_sample
                self.label_map = label_map
        # Shuffle the data
        random.shuffle(data)

        # Splitting the data
        num_samples = len(data)
        train_size = int(training_ratio * num_samples)
        validation_size = int(validation_ratio * num_samples)

        train_data = data[:train_size]
        validation_data = data[train_size:train_size+validation_size]
        test_data = data[train_size+validation_size:]

        # Splitting each dataset into features and labels
        X_train, y_train = zip(*[(sample[:-1], sample[-1]) for sample in train_data])
        X_val, y_val = zip(*[(sample[:-1], sample[-1]) for sample in validation_data])
        X_test, y_test = zip(*[(sample[:-1], sample[-1]) for sample in test_data])

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_inverse_map(self):
        # This will reverse the label map for interpretation
        return {v: k for k, v in self.label_map.items()}