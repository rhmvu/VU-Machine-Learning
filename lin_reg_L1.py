import csv
import numpy as np
import pandas as pd
import matplotlib
import DataPrep

import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LassoCV
from sklearn.model_selection import cross_val_score, train_test_split


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



    # Root mean square error
    def rmse_cv(model):
       
        return  -cross_val_score(model, X_train, y_train,
                                        scoring="neg_mean_squared_error", cv=5)


    # Trying L1 regularization
    parameters = {"fit_intercept":(True,False)} #"n_alphas":(1000,10000)
    model_lasso = LassoCV(alphas=None,
                        cv = 5)
    # model_lasso = GridSearchCV(model_lasso_plain, parameters, cv = 5)
    model_lasso = model_lasso.fit(X_train, y_train)
    rmse_cv(model_lasso).mean()


    # Lasso gives us an alpha of 0.1231, picks some coefficients and gives the rest a 0 value
   # coef = pd.Series(model_lasso.coef_, index=X_train.columns)
    
    # variance score: 1 means perfect prediction
    variance_score = round(model_lasso.score(X_test, y_test), 3)

    if not headless_run:    
        print('Variance score: {}'.format(variance_score))
        
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
        return variance_score


if __name__ == "__main__":
    headless_run = False
    run()
