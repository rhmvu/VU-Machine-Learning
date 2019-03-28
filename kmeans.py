import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import DataPrep

train = DataPrep.prep_data_rico()
target = train.SalePrice
train = train.drop(columns='SalePrice')
numerical_features = train.select_dtypes(exclude=["object"]).columns


X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.25, random_state=0)
#
# stdSc = StandardScaler()
# X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])
# X_test.loc[:, numerical_features] = stdSc.fit_transform(X_test.loc[:, numerical_features])

parameters = {'n_clusters': [1, 2, 4, 6, 8, 10, 12],
              'n_init': [10, 15, 20, 25, 30],
              'max_iter': [300, 400, 500],
              'algorithm': ('auto', 'full', 'elkan'),
              'init': ('k-means++', 'random')}

kmeans = KMeans()
clf = GridSearchCV(kmeans, parameters, 'neg_mean_squared_error', cv=5)
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_estimator_)
print(clf.best_index_)

MSEscore = mean_squared_error(clf.predict(X_test), y_test)
print("MSE", MSEscore)


