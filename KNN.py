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
import result as Result

headless_run = True


def run():
    train = DataPrep.prep_data(headless_run)
    target = train.SalePrice
    train = train.drop(columns='SalePrice')
    numerical_features = train.select_dtypes(exclude=["object"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=0.25, random_state=0)
    #
    # stdSc = StandardScaler()
    # X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])
    # X_test.loc[:, numerical_features] = stdSc.fit_transform(X_test.loc[:, numerical_features])

    # Perform GridSearch. See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
    # for all available parameters.
    parameters = {'n_neighbors': [6, 13, 14, 15, 16, 17], 'algorithm': ('ball_tree', 'kd_tree', 'brute'), 'leaf_size': [
        1, 2, 3, 4], 'weights': ('uniform', 'distance'), 'p': [1, 2]}
    knnr = KNeighborsRegressor()
    clf = GridSearchCV(knnr, parameters, 'neg_mean_squared_error', cv=5)
    clf.fit(X_train, y_train)
    variance_score = round(clf.score(X_test, y_test), 3)

    MSEscore = mean_squared_error(clf.predict(X_test), y_test)
    MAEscore = median_absolute_error(clf.predict(X_test), y_test)
    VarianceScore = explained_variance_score(clf.predict(X_test), y_test)

    if not headless_run:

        print("Mean squared error", MSEscore)
        print("Median absolute error", MAEscore)
        print("Variance score", VarianceScore)
        print('Variance score: {}'.format(variance_score))
        print("CLF BEST: ")
        print(clf.best_score_)
        # print(clf.best_estimator_)
        # print(clf.best_index_)
        print('MSESCORE: ')
        print(MSEscore)
    return Result(variance_score, MSEscore, MAEscore)


if __name__ == "__main__":
    headless_run = False
    run()
