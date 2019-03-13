from sklearn.model_selection import train_test_split
from DataPrep import prep_data_rico
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn import linear_model

# Constants
TRAIN_DATA_PROPORTION = 0.7

# Very basic linear regression model, takes into account all variables

headless_run = True


def run():
    train = prep_data_rico()
    target = train.SalePrice
    train = train.drop(columns='SalePrice')
    numerical_features = train.select_dtypes(exclude=["object"]).columns

    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.25, random_state=0)

    # create linear regression object
    reg = linear_model.LinearRegression()

    # train the model using the training sets
    reg.fit(X_train, y_train)

    # regression coefficients
    if not headless_run:
        print('Coefficients: \n', reg.coef_)

    # variance score: 1 means perfect prediction
    # variance_score = round(reg.score(X_test, y_test), 3)
    MSEscore = mean_squared_error(reg.predict(X_test), y_test)

    if not headless_run:
        print('MSE score: {}'.format(MSEscore))

        # plot for residual error
        plt.style.use('fivethirtyeight')

        # plotting residual errors in training data
        plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
                    color="green", s=10, label='Train data')

        # plotting residual errors in test data
        plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
                    color="blue", s=10, label='Test data')

        # plotting line for zero residual error
        plt.hlines(y=0, xmin=10, xmax=14, linewidth=2)

        plt.legend(loc='upper right')
        plt.title("Residual errors")
        plt.show()
    else:
        return MSEscore


if __name__ == "__main__":
    headless_run = False
    run()
