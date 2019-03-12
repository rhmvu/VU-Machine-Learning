import csv
import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew

from sklearn.model_selection import cross_val_score
from sklearn import svm

# Constants
TRAIN_DATA_PROPORTION = 0.7


# Crappy SVM model

headless_run = True

def run():
    # Data preprocessing
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))

    train.head()
    matplotlib.rcParams['figure.figsize'] = (24.0, 12.0)

    train["SalePrice"] = np.log1p(train["SalePrice"])

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())

    # Split data
    treshold = int(len(train) * TRAIN_DATA_PROPORTION)
    X_train = all_data[0:treshold]
    y_train = train[0:treshold].SalePrice

    X_test = all_data[treshold:len(train)]
    y_test = train[treshold:len(train)].SalePrice

    # Root mean square error
    def rmse_cv(model):
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                        scoring="neg_mean_squared_error", cv=5))
        return rmse

    # Trying SVM using SVR model
    model_svm = svm.SVR(gamma='scale')
    model_svm = model_svm.fit(X_train, y_train)
    rmse_cv(model_svm).mean()  # Why do we need to do this exactly?

    # Coefficients not present since we lack a linear model
    # coef = pd.Series(model_svm.coef_, index=X_train.columns)

    # variance score: 1 means perfect prediction
    variance_score = round(model_svm.score(X_test, y_test), 3)

    if not headless_run:
        print('Variance score: {}'.format(variance_score))

        # Plotting Residuals
        plt.scatter(model_svm.predict(X_train), model_svm.predict(X_train) - y_train,
                    color="green", s=10, label='Train data')

        plt.scatter(model_svm.predict(X_test), model_svm.predict(X_test) - y_test,
                    color="blue", s=10, label='Test data')

        plt.hlines(y=0, xmin=10, xmax=14, linewidth=2)
        plt.legend(loc='upper right')
        plt.title("Residual errors")
        plt.show()
    else:
        return variance_score


if __name__ == "__main__":
    headless_run = False
    run()
