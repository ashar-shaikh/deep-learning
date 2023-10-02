import os
import csv
import random


class MNISTDataProcessor:
    def __init__(self, url, filename):
        self.url = url
        self.filename = filename

    def download_data(self):
        import urllib.request
        urllib.request.urlretrieve(self.url, self.filename)
        print(f"Data downloaded and saved as {self.filename}")

    def process_data(self, training_ratio=0.7, validation_ratio=0.15):
        data = []
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                if row:
                    data.append(row)

        # Shuffle the data
        random.shuffle(data)

        # Splitting the data
        num_samples = len(data)
        train_size = int(training_ratio * num_samples)
        validation_size = int(validation_ratio * num_samples)

        train_data = data[:train_size]
        validation_data = data[train_size:train_size + validation_size]
        test_data = data[train_size + validation_size:]

        # Splitting each dataset into features and labels
        x_train, y_train = zip(*[(sample[1:], sample[0]) for sample in train_data])
        x_val, y_val = zip(*[(sample[1:], sample[0]) for sample in validation_data])
        x_test, y_test = zip(*[(sample[1:], sample[0]) for sample in test_data])

        # Convert string data to integers
        x_train = [list(map(int, x)) for x in x_train]
        y_train = list(map(int, y_train))
        x_val = [list(map(int, x)) for x in x_val]
        y_val = list(map(int, y_val))
        x_test = [list(map(int, x)) for x in x_test]
        y_test = list(map(int, y_test))

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
