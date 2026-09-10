# Model Training Script


import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

data_path = os.path.join("data", "dataset.csv")
model_dir = "model"
model_path = os.path.join(model_dir, "model_l232641.pkl")
target_column = "price"  

learning_rate = 0.1 

def load_data(path):
    print("Loading dataset from ",path)
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def train_model(df):
    x = df.drop(columns=[target_column])
    x = x.select_dtypes(include=["number"])  #keep numeric columns only
    y = df[target_column]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    print("Model trained. Test MSE: ", mse)

    return model

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print("Model saved to ", path)


dataframe = load_data(data_path)
trained_model = train_model(dataframe)
save_model(trained_model, model_path)