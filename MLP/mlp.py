import random
import math


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Initialize weights and biases with random values
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]

        self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(output_size)]

        self.learning_rate = learning_rate
        self.train_data = ""
        self.validation_data = ""
        self.testing_data = ""
        self.data_processed = False

        self.hidden_input = ""
        self.hidden_output = ""
        self.final_input = ""
        self.final_output = ""

    def load_data(self, train_data, validation_data, testing_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.testing_data = testing_data
        self.data_processed = True

    def sigmoid(self, x):
        # Sigmoid activation function
        if x < 0:
            z = math.exp(x)
            return z / (1 + z)
        else:
            return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function
        return x * (1 - x)

    def softmax(self, x):
        # Softmax' activation function for the output layer
        exps = [math.exp(i - max(x)) for i in x]
        return [exp / sum(exps) for exp in exps]

    def forward(self, input_data):
        # Hidden layer
        self.hidden_input = [sum([i * w for i, w in zip(input_data, w_row)]) + b for w_row, b in
                             zip(self.weights_input_hidden, self.bias_hidden)]
        self.hidden_output = [self.sigmoid(i) for i in self.hidden_input]

        # Output layer
        self.final_input = [sum([h * w for h, w in zip(self.hidden_output, w_row)]) + b for w_row, b in
                            zip(self.weights_hidden_output, self.bias_output)]
        self.final_output = self.softmax(self.final_input)

        return self.final_output

    def backward(self, input_data, target_output):
        # Calculate the error for the output layer
        output_error = [t - o for t, o in zip(target_output, self.final_output)]

        # Output layer gradient
        d_output = output_error  # Since the derivative of softmax with cross-entropy is the error itself

        # Hidden layer gradient
        d_hidden = [sum([d_o * w for d_o, w in zip(d_output, w_row)]) * self.sigmoid_derivative(h) for w_row, h in
                    zip(zip(*self.weights_hidden_output),
                        self.hidden_output)]  # Transpose weights_hidden_output for correct matrix multiplication

        # Update weights and biases for the output layer
        for i in range(len(self.weights_hidden_output)):
            for j in range(len(self.weights_hidden_output[i])):
                self.weights_hidden_output[i][j] += self.learning_rate * d_output[i] * self.hidden_output[j]
            self.bias_output[i] += self.learning_rate * d_output[i]

        # Update weights and biases for the hidden layer
        for i in range(len(self.weights_input_hidden)):
            for j in range(len(self.weights_input_hidden[i])):
                self.weights_input_hidden[i][j] += self.learning_rate * d_hidden[i] * input_data[j]
            self.bias_hidden[i] += self.learning_rate * d_hidden[i]

    def train(self, epochs):
        if not self.data_processed:
            raise ValueError("Data not loaded. Please load data before training.")

        training_inputs, training_outputs = self.train_data
        validation_inputs, validation_outputs = self.validation_data

        for epoch in range(epochs):
            total_correct = 0
            for idx, (input_data, target_output) in enumerate(zip(training_inputs, training_outputs)):
                # Forward pass
                output = self.forward(input_data)
                predicted_label = output.index(max(output))

                # Check if the labels are one-hot encoded or direct integers
                if isinstance(target_output, list):
                    actual_label = target_output.index(1)
                else:
                    actual_label = target_output

                if predicted_label == actual_label:
                    total_correct += 1
                # Backward pass and weight update
                self.backward(input_data, self.one_hot_encode(target_output, len(output)))
                # Print progress every 5000 iterations
                if (idx + 1) % 5000 == 0:
                    current_accuracy = total_correct / (idx + 1)
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Iteration {idx + 1}/{len(training_inputs)}, Current Accuracy: {current_accuracy:.4f}")

            accuracy = total_correct / len(training_inputs)
            validation_accuracy = self.validate()
            print(
                f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {accuracy:.4f}, Validation Accuracy: {validation_accuracy:.4f}")
    def one_hot_encode(self, label, num_classes):
        """Helper function to one-hot encode a label."""
        encoding = [0] * num_classes
        encoding[label] = 1
        return encoding

    def validate(self):
        if not self.data_processed:
            raise ValueError("Data not loaded. Please load data before validation.")

        validation_inputs, validation_outputs = self.validation_data
        total_correct = 0
        for input_data, target_output in zip(validation_inputs, validation_outputs):
            output = self.forward(input_data)
            predicted_label = output.index(max(output))

            # Check if the labels are one-hot encoded or direct integers
            if isinstance(target_output, list):
                actual_label = target_output.index(1)
            else:
                actual_label = target_output

            if predicted_label == actual_label:
                total_correct += 1
        return total_correct / len(validation_inputs)

    def predict(self, input_data):
        output = self.forward(input_data)
        return output.index(max(output))

    def test(self):
        if not self.data_processed:
            raise ValueError("Data not loaded. Please load data before testing.")

        testing_inputs, testing_outputs = self.testing_data
        total_correct = 0
        for input_data, target_output in zip(testing_inputs, testing_outputs):
            output = self.forward(input_data)
            predicted_label = output.index(max(output))

            # Check if the labels are one-hot encoded or direct integers
            if isinstance(target_output, list):
                actual_label = target_output.index(1)
            else:
                actual_label = target_output

            if predicted_label == actual_label:
                total_correct += 1
        return total_correct / len(testing_inputs)