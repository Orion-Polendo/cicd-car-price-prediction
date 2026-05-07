# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    
    # -------- WRITE YOUR CODE HERE --------
    
    # Step 1: Define arguments for train data, test data, model output, and RandomForest hyperparameters. Specify their types and defaults.  
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--criterion", type=str, default="gini", help="The function to measure the quality of a split")
    parser.add_argument("--max_depth", type=int, default=None,
        help="The maximum depth of the tree. If None, then nodes are expanded until all leaves contain < min_smaples_split samples."),
    parser.add_argument("--model_output", type=str, help="Path of output model")
    args = parser.parse_args()


    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # -------- WRITE YOUR CODE HERE --------

    # Step 2: Read the train and test datasets from the provided paths using pandas. Replace '_______' with appropriate file paths and methods.  
    # Step 3: Split the data into features (X) and target (y) for both train and test datasets. Specify the target column name.  
    # Step 4: Initialize the RandomForest Regressor with specified hyperparameters, and train the model using the training data.  
    # Step 5: Log model hyperparameters like 'n_estimators' and 'max_depth' for tracking purposes in MLflow.  
    # Step 6: Predict target values on the test dataset using the trained model, and calculate the mean squared error.  
    # Step 7: Log the MSE metric in MLflow for model evaluation, and save the trained model to the specified output path.  

    # Load the Train and Test Datasets
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Establish y_train
    y_train = train_df['price'].values # Target Variable

    # Establish X_train
    X_train = train_df.drop("price", axis=1).values

    # Establish y_test
    y_test = test_df["price"].values

    # Establish X_test
    X_test = test_df.drop("price", axis=1).values

    # Intialize a Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth = args.max_depth,
        random_state=42
    )

    # Train the Model
    rf_model = rf_model.fit(
        X_train,
        y_train
    )

    # Log Model Hyperparameters
    mlflow.log_param("model", "Random Forest Regressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Predict
    y_pred = rf_model.predict(X_test)

    # Compute and Log the MSE for Testing Data
    mse = mean_squared_error(
        y_test,
        y_pred
    )
    print("Mean Squared Error of Random Forest Regressor on test set: {:.2f}".format(mse))
    mlflow.log_metric("MSE", float(mse))

    # Output the trained model
    mlflow.sklearn.save_model(
        sk_model = rf_model,
        path = args.model_output
    )
    
    # Ending the MLflow experiment run
    mlflow.end_run()

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

