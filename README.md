# Email Classification

This project is an implementation of an LSTM (Long Short-Term Memory) neural network for email classification using PyTorch. The goal is to classify emails as either spam or ham (not spam) based on their content.

## Data

The dataset used in this project is obtained from Kaggle and can be downloaded from the following link: https://www.kaggle.com/datasets/datatattle/email-classification-nlp

The dataset consists of two CSV files: `SMS_train.csv` and `SMS_test.csv`. Each file contains the following columns:

- `Message_body`: The text content of the email.
- `Label`: The label indicating whether the email is spam (`spam`) or ham (`ham`).

The dataset files should be placed in the `data` directory.

## Dependencies

This project requires the following Python libraries:

- pandas
- scikit-learn
- torch

You can install them using pip:

```
pip install pandas scikit-learn torch
```

## File Structure

- `data/link.txt`: Contains the link to the Kaggle dataset used in this project.
- `data_utils.py`: Contains utility functions for loading and preprocessing the data.
- `main.py`: The main entry point of the application.
- `model.py`: Defines the LSTM model architecture.
- `predict.py`: Contains a function for making predictions on new text data.
- `train.py`: Contains functions for training the LSTM model.

## Usage

1. Download the dataset from the provided Kaggle link and place the `SMS_train.csv` and `SMS_test.csv` files in the `data` directory.
2. Run `main.py` to train the model and make a prediction on a sample text.

The trained model will be saved in the `models` directory as `first_model.pth`.

## Code Overview

### `data_utils.py`

- `load_data()`: Loads the training and test data from the CSV files.
- `preprocess_data()`: Preprocesses the data by encoding the labels, creating bag-of-words vectors, and splitting the data into input and target variables.

### `main.py`

- Imports necessary modules and functions.
- Loads and preprocesses the data using `load_data` and `preprocess_data` from `data_utils.py`.
- Creates an instance of the LSTM model from `model.py`.
- Trains the model using `train_model` from `train.py`.
- Makes a prediction on a sample text using `predict_text` from `predict.py`.
- Saves the trained model to the `models` directory.

### `model.py`

- Defines the LSTM model architecture using PyTorch's `nn.Module`.

### `predict.py`

- `predict_text()`: Converts the input text into a bag-of-words vector and makes a prediction using the trained model.

### `train.py`

- `SequencesDataset`: A custom PyTorch dataset for loading and batching the training data.
- `train_model()`: Trains the LSTM model using the provided training data and hyperparameters.

## Note

This README provides a high-level overview of the project structure and functionality. For more detailed information and implementation details, please refer to the respective Python files.
