
# imports
import os
import mlflow
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# define functions
def main(args):
    # enable auto logging
    mlflow.autolog()

    # setup parameters
    params = {
        "fit_intercept": args.fit_intercept,
        "normalize": args.normalize,
        "positive": args.positive,
    }

    # read in data
    df = pd.read_csv(args.data)

    # process data
    X_train, X_test, y_train, y_test = process_data(df, args.random_state)

    # train model
    model = train_model(params, X_train, X_test, y_train, y_test)

    # Stop Logging
    # mlflow.end_run()

def process_data(df, random_state):
    # split dataframe into X and y
    X = df.drop(["target"], axis=1)
    y = df["target"]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # return splits and encoder
    return X_train, X_test, y_train, y_test

def train_model(params, X_train, X_test, y_train, y_test):
    # train model
    model = LinearRegression(**params)
    model = model.fit(X_train, y_train)

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=model,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )

    # return model
    return model

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--data", type=str)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--fit_intercept", type=bool, default=True)
    parser.add_argument("--normalize", type=bool, default=False)
    parser.add_argument("--positive", type=bool, default=False)
    parser.add_argument("--registered_model_name", type=str, help="model name")

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
