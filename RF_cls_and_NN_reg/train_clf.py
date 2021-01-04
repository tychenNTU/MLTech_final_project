import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score, GridSearchCV
import joblib
import datautil
from sklearn.metrics import classification_report
import pickle as pkl

with open("../preprocess/processed_train.pkl", "rb") as f:
    global train_x, train_y
    train_x, train_y = pkl.load(f)

# build the random forest classifier
rf_cls = RandomForestClassifier(random_state=42, n_jobs=-1)
kfolds = 5 # 5-cross validation
split = KFold(kfolds, shuffle=True, random_state=42)

# tune model 
# rf_parameters = {"max_depth": [10,13],
#                  "n_estimators": [10,100,500],
#                  "min_samples_split": [2,5]}
# rf_model = RandomForestClassifier()
# rf_cv_model = GridSearchCV(rf_model,
#                            rf_parameters,
#                            cv = 10,
#                            n_jobs = -1,
#                            verbose = 2)

# rf_cv_model.fit(X_transformed, y_train_full)
# print('Best parameters: ' + str(rf_cv_model.best_params_))

rf_tuned = RandomForestClassifier(max_depth = 13,
                                  min_samples_split = 5,
                                  n_estimators = 500)

print('Model: Random Forest Tuned\n')

rf_tuned.fit(train_x, train_y)

# get cross validation score for the model
cv_results = cross_val_score(rf_tuned, X_transformed, y_train_full, 
                            cv=split, scoring="accuracy", verbose=3, n_jobs=-1)
min_score = round(min(cv_results), 4) # round to 4 decimal precision
max_score = round(max(cv_results), 4)
mean_score = round(np.mean(cv_results), 4)
std_dev = round(np.std(cv_results), 4)
print(f"cross validation accuracy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")

joblib.dump(rf_tuned, "rf_cls.model")
