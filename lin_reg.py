import numpy as np
import pandas as pd
from DataPrep import prep_data_rico
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


# Constants
TRAIN_DATA_PROPORTION = 0.7

# Somewhat more advanced regression, performs regularization to eliminate some variables that are not necessary

headless_run = True


def run():
    train = prep_data_rico()
    target = train.SalePrice
    train = train.drop(columns='SalePrice')
    numerical_features = train.select_dtypes(exclude=["object"]).columns

    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.25, random_state=0)

    # Root mean square error
    def rmse_cv(model):
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train,
                                        scoring="neg_mean_squared_error", cv=5))
        return rmse

    # Trying L1 regularization
    model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005], cv=5).fit(X_train, y_train)
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
