import numpy as np
import pandas as pd
import matplotlib
import DataPrep

import matplotlib.pyplot as plt
from scipy.stats import skew

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

# Constants
TRAIN_DATA_PROPORTION = 0.7

# Somewhat more advanced regression, performs regularization to eliminate some variables that are not necessary

headless_run = True

def run():
    # Data preprocessing
    train = DataPrep.prep_data(headless_run)
    # Scale data: https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use

    target = train.SalePrice
    train = train.drop(columns='SalePrice')

    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=0.25, random_state=0)



    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.25, random_state=0)

    # Root mean square error
    def rmse_cv(model):
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                        scoring="neg_mean_squared_error", cv=5))
        return rmse

    # Trying L1 regularization
    model_lasso = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(X_train, y_train)

    rmse_cv(model_lasso).mean()

    # Lasso gives us an alpha of 0.1231, picks some coefficients and gives the rest a 0 value
    coef = pd.Series(model_lasso.coef_, index=X_train.columns)
    
    # variance score: 1 means perfect prediction
    variance_score = round(model_lasso.score(X_test, y_test), 3)
    MSEscore = mean_squared_error(model_lasso.predict(X_test), y_test)

    if not headless_run:
        print('MSE score: {}'.format(MSEscore))
        
        # Plotting Residuals

        plt.scatter(model_lasso.predict(X_train), model_lasso.predict(X_train) - y_train,
                    color="green", s=10, label='Train data')

        plt.scatter(model_lasso.predict(X_test), model_lasso.predict(X_test) - y_test,
                    color="blue", s=10, label='Test data')

        plt.hlines(y=0, xmin=10, xmax=14, linewidth=2)

        plt.legend(loc='upper right')
        plt.title("Residual errors")
        plt.show()
    else:
        return MSEscore


if __name__ == "__main__":
    headless_run = False
    run()
