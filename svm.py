import csv
import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer


import DataPrep

# Constants
TEST_SIZE = 0.25


# Crappy SVM model

headless_run = True


def run():
    # Data preprocessing

    train = DataPrep.prep_data_rico(headless_run)
    # Scale data: https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
    train = DataPrep.normalize(train)

    target = train.SalePrice
    train = train.drop(columns='SalePrice')

    # print(train)
    numerical_features = train.select_dtypes(exclude=["object"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=TEST_SIZE, random_state=0)

    # Root mean square error

    def rmse_cv(model):
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                        scoring="neg_mean_squared_error", cv=5))
        return rmse

    # Trying SVM using SVR model: https://scikit-learn.org/stable/modules/svm.html
    model_svm = svm.SVR(gamma='scale', cache_size=1000)
    parameters = {'kernel': ['rbf', 'sigmoid', 'linear'], 'C': [0.8, 0.9, 1]}

    #model_svm = GridSearchCV(model_svm, parameters, 'neg_mean_squared_error', cv=5)
    model_svm = model_svm.fit(X_train, y_train)
    rmse_cv(model_svm).mean()  # Why do we need to do this exactly?

    # Coefficients not present since we lack a linear model
    # coef = pd.Series(model_svm.coef_, index=X_train.columns)

    # variance score: 1 means perfect prediction
    variance_score = round(model_svm.score(X_test, y_test), 3)

    if not headless_run:
        print('Variance score: {}'.format(variance_score))

        MSEscore = mean_squared_error(model_svm.predict(X_test), y_test)
        print(MSEscore)

        # print(model_svm.best_score_)
        # print(model_svm.best_estimator_)
        # print(model_svm.best_index_)

        # Plotting Residuals
        plt.scatter(model_svm.predict(X_train), model_svm.predict(X_train) - y_train,
                    color="green", s=10, label='Train data')

        plt.scatter(model_svm.predict(X_test), model_svm.predict(X_test) - y_test,
                    color="blue", s=10, label='Test data')

        plt.hlines(y=0, xmin=0, xmax=1, linewidth=2)
        plt.legend(loc='upper right')
        plt.title("Residual errors")
        plt.show()
    else:
        return variance_score


if __name__ == "__main__":
    headless_run = False
    run()
