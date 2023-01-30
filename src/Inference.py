#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

with open('trained_model.pkl', 'rb') as f:
    rf = pickle.load(f)


def predict(model, new_data):
    predictions = model.predict(new_data)
    return predictions


df = pd.read_csv("../data/term-deposit-marketing-2020.csv")

    
# Data preprocessing
from preprocess import preprocess_data
processed_df = preprocess_data(df)

Y = processed_df['y_yes']
X = processed_df.drop(['y_yes'], axis=1)

ints = ['age', 'balance', 'duration', 'day']
ct = ColumnTransformer([
        ('somename', StandardScaler(), ints)
    ], remainder='passthrough')

X_stan = ct.fit_transform(X)
X_stan = pd.DataFrame(X_stan, columns=X.columns)



predictions = predict(rf, X_stan)
predictions

