# Base on stock_pred.py

from datetime import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, SimpleRNN

from xgboost import XGBRegressor
import joblib

# check GPU
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
TIME_STEP = 60


def create_train_dataset(df, features=["Close"], time_step=60):
    features = sorted(features)
    df["Date"] = pd.to_datetime(df.Date,format="%Y-%m-%d")
    df.index = df["Date"]
    df = df.sort_index(ascending=True,axis=0)
    df = df.reset_index(drop=True)

    # Scaler
    dct_scaler = {}
    for i,feat in enumerate(features):
        dct_scaler[feat] = MinMaxScaler(feature_range=(0,1))
        df[feat] = dct_scaler[feat].fit_transform(df[feat].values.reshape(-1,1)).reshape(-1)

    train_df_root, test_df_root = df.iloc[0:987, ], df.iloc[987-time_step:, ]
    train_df, test_df = train_df_root[features], test_df_root[features]

    X_train = np.zeros((train_df.shape[0]-time_step, time_step, len(features)))
    y_train = np.zeros((train_df.shape[0]-time_step, ))
    for i in range(time_step, train_df.shape[0]):
        X_train[i-time_step] = train_df.values[i-time_step:i,:]
        y_train[i-time_step] = train_df.values[i,0]
    
    X_test = np.zeros((test_df.shape[0]-time_step, time_step, len(features)))
    y_test = np.zeros((test_df.shape[0]-time_step, ))
    for i in range(time_step, test_df.shape[0]):
        X_test[i-time_step] = test_df.values[i-time_step:i,:]
        y_test[i-time_step] = test_df.values[i,0]
    return X_train, y_train, X_test, y_test, dct_scaler, train_df_root, test_df_root


def build_model_lstm(input_shape):
    model=Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(
        loss="mean_squared_error",
        optimizer="adam")
    return model

def train_model_lstm(X_train, y_train, output_path="saved_lstm_model.h5"):
    lstm_model = build_model_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))

    lstm_model.fit(
        X_train, 
        y_train,
        epochs=10,
        batch_size=32,
        verbose=2)

    lstm_model.save(output_path)
    return lstm_model



def build_model_rnn(input_shape):
    model=Sequential()
    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=input_shape))
    model.add(SimpleRNN(units=50, return_sequences=True))
    model.add(SimpleRNN(units=50))
    model.add(Dense(units=1))

    model.compile(
        loss="mean_squared_error",
        optimizer="adam")
    return model

def train_model_rnn(X_train, y_train, output_path="saved_rnn_model.h5"):
    rnn_model = build_model_rnn(input_shape=(X_train.shape[1], X_train.shape[2]))

    rnn_model.fit(
        X_train, 
        y_train,
        epochs=10,
        batch_size=32,
        verbose=2)

    rnn_model.save(output_path)
    return rnn_model



def build_model_xgboost():
    model = XGBRegressor(
        n_estimators=100,
        objective="reg:squarederror",
        gamma=0.01,
        learning_rate=0.01,
        max_depth=4,
        random_state=42,
        subsample=1, 
        verbosity=2,
        seed=132,
    )
    return model

def train_model_xgboost(X_train, y_train, output_path="saved_xgboost_model.joblib"):
    xgboost_model = build_model_xgboost()
    print(xgboost_model)
    xgboost_model.fit(X_train.reshape((X_train.shape[0], -1)), y_train)
    joblib.dump(xgboost_model, output_path)
    return xgboost_model



def train(X_train, y_train, method="LSTM", output_path="model_output/saved_lstm_model.h5"):
    if method=="LSTM":
        print("Training model LSTM ...")
        lstm = train_model_lstm(X_train, y_train, output_path)
        return lstm

    elif method=="RNN":
        print("Training model RNN ...")
        output_path = "model_output/saved_rnn_model.h5"
        rnn = train_model_rnn(X_train, y_train, output_path)
        return rnn


    elif method=="XGBOOST":
        print("Training model Xgboost ...")
        output_path = "model_output/saved_xgboost_model.joblib"
        xgboost = train_model_xgboost(X_train, y_train, output_path)
        return xgboost



if __name__ == "__main__":
    df = pd.read_csv("NSE-TATA.csv")
    features = ['Close']
    X_train, y_train, X_test, y_test, dct_scaler, train_df, test_df = create_train_dataset(df, features=features, time_step=TIME_STEP)

    lstm = train(X_train, y_train, method="LSTM", output_path="model_output/saved_lstm_model.h5")

    rnn = train(X_train, y_train, method="RNN", output_path="model_output/saved_rnn_model.h5")

    xgboost = train(X_train, y_train, method="XGBOOST", output_path="model_output/saved_xgboost_model.joblib")