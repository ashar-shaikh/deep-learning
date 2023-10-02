from mnist_data_processor import MNISTDataProcessor as mdp
from mlp import MLP

URL = "https://path_to_mnist_csv/mnist.csv"  # Replace with the actual URL if you have one
FILE_NAME = 'mnist.csv'

LEARNING_RATE = 0.01
TRAINING_DATA = 0.7
VAL_DATA = 0.15
EPOCHS = 10

try:
    processor = mdp(URL, FILE_NAME)

    # Download data
    # processor.download_data()

    # Process data
    train_data, val_data, test_data = processor.process_data(TRAINING_DATA, VAL_DATA)

    # Get a sample from the train_data for the MLP initialization
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data

    input_size = len(x_train[0])
    hidden_size = 128  # You can adjust this
    output_size = 10  # For 10 digits

    mlp = MLP(input_size, hidden_size, output_size, LEARNING_RATE)

    mlp.load_data(train_data, val_data, test_data)

    mlp.train(EPOCHS)

    print(f"Validation Accuracy:  {mlp.validate() * 100}%")
    print(f"Testing Accuracy: {mlp.test() * 100}%")

    # To get a prediction on new data (a sample from the test set for demonstration):
    sample_data = x_test[0]
    predicted_class = mlp.predict(sample_data)
    print(f"The predicted digit for the sample data is: {predicted_class}")

except Exception as e:
    print("Error:", e)
