# Churn-Modelling

# Customer Churn Prediction with Artificial Neural Network

This repository contains the code for building an Artificial Neural Network (ANN) to predict customer churn. The model is trained on a dataset of customer information and aims to identify customers who are likely to leave a service or product.

## Project Overview

The project is structured as follows:

1.  **Data Loading and Preparation:**
    *   Loads data from a CSV file named `Churn_Modelling.csv` using `pandas`.
    *   Extracts features (`X`) and target variable (`y`).
2.  **Data Preprocessing:**
    *   **Categorical Encoding:**
        *   Uses `LabelEncoder` to encode gender into numerical values.
        *   Uses `OneHotEncoder` to encode categorical features such as country into binary columns.
    *   **Train/Test Split:** Splits the data into training and test sets using `train_test_split`.
    *   **Feature Scaling:** Scales the data using `StandardScaler` to normalize feature values, improving training efficiency.
3.  **Building the Artificial Neural Network:**
    *   **Initialization:** A sequential model is initialized.
    *   **Hidden Layers:** Two fully connected dense layers with ReLU activation are added, each with 6 units.
    *   **Output Layer:** An output layer with a sigmoid activation is used for binary classification.
4.  **Training the ANN:**
    *   **Compilation:** The model is compiled using the Adam optimizer, binary cross-entropy loss function, and accuracy metrics.
    *   **Training:** The model is trained on the preprocessed training data for 30 epochs with a batch size of 32.
5.  **Making a Single Prediction:**
    *   The model predicts the churn probability for a sample customer, using preprocessed sample data, and prints the prediction.
6.  **Accuracy Calculation:**
    *   The model makes predictions on the test set.
    *   Predictions are converted to binary values (0 or 1) using a threshold of 0.5.
    *   The model's accuracy is calculated using `accuracy_score` and printed.

## Libraries Used

*   `pandas`
*   `scikit-learn`
*   `tensorflow`

## How to Use

1.  Make sure you have the libraries listed installed (pandas, scikit-learn and tensorflow).
2.  Place the `Churn_Modelling.csv` file in the same directory as the code.
3.  Run the Python script to train the model and see the test set accuracy.
4.  You can change the input data to `single_observation` to experiment with different customer information.

## Notes

*   The provided dataset `Churn_Modelling.csv` can be replaced by other datasets.
*   The ANN architecture is a good starting point, but more layers and different activations can be tested.
*   The model can be further improved by fine-tuning hyperparameters.
