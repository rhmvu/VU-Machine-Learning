import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import DataPrep

train = DataPrep.prep_data()
target = train.SalePrice
train = train.drop(columns='SalePrice')
numerical_features = train.select_dtypes(exclude = ["object"]).columns


X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.2, random_state = 0)
#
# stdSc = StandardScaler()
# X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])
# X_test.loc[:, numerical_features] = stdSc.fit_transform(X_test.loc[:, numerical_features])

# Perform GridSearch. See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
# for all available parameters.
parameters = {'n_neighbors':[12,13,4,15], 'algorithm':('ball_tree', 'kd_tree', 'brute'), 'leaf_size':[1,50,60], 'weights':('uniform', 'distance'), 'p':[1,2]}
knnr = KNeighborsRegressor()
clf = GridSearchCV(knnr, parameters, 'neg_mean_squared_error', cv=3)
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_estimator_)
print(clf.best_index_)

MSEscore = mean_squared_error(clf.predict(X_test),y_test)
print(MSEscore)






