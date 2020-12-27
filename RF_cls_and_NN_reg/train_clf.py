import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
import joblib
import datautil

full_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../train.csv'))
full_data = full_data[(full_data['adr'] < 1000) & (full_data['adr'] > -100)] # remove outliers

# get the preprocessor and the default training features
preprocessor, features_spec = datautil.get_the_data_preprocessor()

# split data into input and labeled
X_train_full_raw = full_data[features_spec]
y_train_full = np.array(full_data['is_canceled'])
X_transformed = preprocessor.fit_transform(X_train_full_raw)

# build the random forest classifier
rf_cls = RandomForestClassifier(random_state=42, n_jobs=-1)
kfolds = 5 # 5-cross validation
split = KFold(kfolds, shuffle=True, random_state=42)


# get cross validation score for the model
cv_results = cross_val_score(rf_cls, X_transformed, y_train_full, 
                            cv=split, scoring="neg_mean_squared_error")
min_score = round(min(cv_results), 4) # round to 4 decimal precision
max_score = round(max(cv_results), 4)
mean_score = round(np.mean(cv_results), 4)
std_dev = round(np.std(cv_results), 4)
print(f"cross validation accuracy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")
rf_cls.fit(X_transformed, y_train_full)
joblib.dump(rf_cls, "rf_cls.model")