#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from scipy import stats

import pickle


from sklearn.ensemble import RandomForestClassifier

# Metrics to evaluate the model
from sklearn.metrics import (
    f1_score,
    accuracy_score,   
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,

    classification_report,
    precision_recall_curve
)


df = pd.read_csv("../data/term-deposit-marketing-2020.csv")
df.head()

    
# Data preprocessing
from preprocess import preprocess_data
processed_df = preprocess_data(df)

# Outlier Detection
## Lets apply outlier detection using 3 Z scores, just for class 0
# calculate the z-score of each column
z = np.abs(stats.zscore(processed_df))

# drop the rows that have a z-score greater than 3 for only class '0'
processed_df = processed_df[(z < 3).all(axis=1) | (processed_df['y_yes'] != 0)]

Y = processed_df['y_yes']
X = processed_df.drop(['y_yes'], axis=1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=12)

ints = ['age', 'balance', 'duration', 'day']
ct = ColumnTransformer([
        ('somename', StandardScaler(), ints)
    ], remainder='passthrough')

X_train_stan = ct.fit_transform(X_train)
X_train_stan = pd.DataFrame(X_train_stan, columns=X.columns)

X_test_stan = ct.transform(X_test)
X_test_stan = pd.DataFrame(X_test_stan, columns=X.columns)


#Random Forest
n_estimators = [75, 100, 125]
criterion=['entropy']
min_samples_split = [11, 12, 14]
class_weight=["balanced"]
rf = RandomForestClassifier()
parameters = dict(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, class_weight=class_weight)
grid_search_rf = GridSearchCV(estimator = rf,
                           param_grid = parameters,
                           scoring = 'f1',
                           cv = StratifiedKFold(n_splits=5),
                           n_jobs = -1)
grid_search_rf = grid_search_rf.fit(X_train_stan, Y_train)


# Evaluation
##creating metric function 
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

prd_test_rf = grid_search_rf.predict(X_test_stan)
metrics_score(Y_test, prd_test_rf)