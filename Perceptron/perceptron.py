import random

class Perceptron():
    def __init__(self, num_features, bias, learning_rate):
        self.weight = [random.uniform(-1, 1) for _ in range(num_features)]
        # self.weight = [0 for _ in range(num_features)]

        self.bias = bias
        self.learning_rate = learning_rate
        self.data_processed = False

    def load_data(self, train_data, validation_data, testing_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.testing_data = testing_data
        self.data_processed = True
        return True

    def forward(self, input):
        z = sum([i*w for i, w in zip(input, self.weight)]) + self.bias
        return 1 if z > 0 else 0

    def train(self, epochs):
        if not self.data_processed:
            print("Data not loaded!")
            return

        for epoch in range(epochs):
            total_errors = 0
            for X, y in zip(*self.train_data):
                prediction = self.forward(X)
                error = y - prediction
                
                # If error is not 0 (i.e., if the prediction was wrong), accumulate error
                if error != 0:
                    total_errors += 1

                # Update weights and bias
                for idx, x in enumerate(X):
                    self.weight[idx] += self.learning_rate * error * x
                self.bias += self.learning_rate * error
            
            # Optional: Print error for each epoch to monitor convergence
            
            print(f"Epoch {epoch+1}/{epochs}, Total Misclassifications: {total_errors}")
            
            # If no errors, perceptron converged, so break
            if total_errors == 0:
                break

        accuracy = (len(self.train_data[0]) - total_errors) / len(self.train_data[0])
        print(f"Training accuracy after {epochs} epochs: {accuracy * 100:.2f}%,  Final Bias: {self.bias}")
        return total_errors


    def validate(self):
        correct_predictions = 0
        for X, y in zip(*self.validation_data):
            prediction = self.forward(X)
            if prediction == y:
                correct_predictions += 1
        return correct_predictions / len(self.validation_data[0])

    def test(self):
        correct_predictions = 0
        for X, y in zip(*self.testing_data):
            prediction = self.forward(X)
            if prediction == y:
                correct_predictions += 1
        return correct_predictions / len(self.testing_data[0])

    def predict(self, input):
        return self.forward(input)
