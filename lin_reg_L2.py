import csv
import numpy as np
import pandas as pd
import matplotlib
import DataPrep
from result import Result
import matplotlib.pyplot as plt
from scipy.stats import skew

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error


# Somewhat more advanced regression, performs L1 regularization to eliminate some variables that are not necessary


headless_run = True


def run():
    # Data preprocessing
    train = DataPrep.prep_data(headless_run)
    # Scale data: https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use

    target = train.SalePrice
    train = train.drop(columns='SalePrice')

    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=0.25, random_state=0)

    # Trying L2 regularization
    clf = RidgeCV(cv=5).fit(X_train, y_train)
    # print(rmse_cv(clf).mean())

    # Lasso gives us an alpha of 0.1231, picks some coefficients and gives the rest a 0 value
    coef = pd.Series(clf.coef_, index=X_train.columns)

    # variance score: 1 means perfect prediction
   # variance score: 1 means perfect prediction
    variance_score = clf.score(X_test, y_test)

    MSEscore = mean_squared_error(clf.predict(X_test), y_test)

    MAEscore = median_absolute_error(clf.predict(X_test), y_test)

    if not headless_run:
        print('Variance score: {}'.format(variance_score))
        # print("CLF best: {}".format(clf.best_score_)) grid search only
        print('MSE score: {}'.format(MSEscore))
        print('MAE score: {}'.format(MAEscore))

        # Plotting Residuals

        plt.scatter(clf.predict(X_train), clf.predict(X_train) - y_train,
                    color="green", s=10, label='Train data')

        plt.scatter(clf.predict(X_test), clf.predict(X_test) - y_test,
                    color="blue", s=10, label='Test data')

        plt.hlines(y=0, xmin=10, xmax=14, linewidth=2)

        plt.legend(loc='upper right')
        plt.title("Residual errors")
        plt.show()
    else:
        return Result(variance_score, MSEscore, MAEscore)


if __name__ == "__main__":
    headless_run = False
    run()
