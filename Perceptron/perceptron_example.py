from dataProcessor import DataProcessor as dp
from perceptron import Perceptron as P

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
FILE_NAME = "Perceptron/breast_cancer.csv"

BIAS = 1.357
LEARNING_RATE =  0.022
TRAINING_DATA = 0.6
VAL_DATA = 0.2
EPOCHS = 64
HAS_CODE = True # Removes the first column from the dataset. 
try:
    processor = dp(URL, FILE_NAME)

    # Download data
    processor.download_data()

    # Process data
    train_data, val_data, test_data = processor.process_data(0.6, 0.2, HAS_CODE)

    # Get a sample from the train_data for the Perceptron initialization
    X_train, y_train = train_data
    sample_data_len = len(X_train[0])

    p = P(sample_data_len, BIAS, LEARNING_RATE)

    p.load_data(train_data, val_data, test_data)

    print("Total Errors:", p.train(EPOCHS))
    print("Learning Rate:" + str(LEARNING_RATE))
    print("Bias:" + str(BIAS))

    print(f"Validation Rate:  {p.validate() * 100}%")
    print(f"Testing Rate: {p.test() * 100}%")

    # To get a prediction on new data:

    # Clump_thickness: 5
    # Uniformity_of_cell_size: 7
    # Uniformity_of_cell_shape: 6
    # Marginal_adhesion: 4
    # Single_epithelial_cell_size: 3
    # Bare_nuclei: 1
    # Bland_chromatin: 8
    # Normal_nucleoli: 2
    # Mitoses: 1
    # High chances of Malignancy
    new_data = [5, 7, 6, 4, 3, 1, 8, 2, 1]
    predicted_class = p.predict(new_data)
    inverse_map = processor.get_inverse_map()
    # (2 for benign, 4 for malignant)
    labels = {
        "2": "Benign",
        "4": "Malignant"
    }
    print(f"The predicted class for the new data is: {labels[inverse_map[predicted_class]]}")
except Exception as e:
    print("Error:", e)