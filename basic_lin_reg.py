from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, median_absolute_error

from sklearn import linear_model
from sklearn.model_selection import cross_val_score, train_test_split
import DataPrep
from sklearn.metrics import mean_squared_error, median_absolute_error, explained_variance_score, r2_score

# Very basic linear clfression model, takes into account all variables

headless_run = True


def run():
    # Data preprocessing
    train = DataPrep.prep_data(headless_run)
    # Scale data: https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use

    target = train.SalePrice
    train = train.drop(columns='SalePrice')

    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=0.25, random_state=0)

    # create linear clfression object
    clf = linear_model.LinearRegression()

    # train the model using the training sets
    clf.fit(X_train, y_train)

    # clfression coefficients
    if not headless_run:
        print('Coefficients: \n', clf.coef_)

    # Metrics
    variance_score = clf.score(X_test, y_test)
    MSEscore = mean_squared_error(clf.predict(X_test), y_test)
    MAEscore = median_absolute_error(clf.predict(X_test), y_test)
    R2score = r2_score(clf.predict(X_test), y_test)

    if not headless_run:
        print('Variance score: {}'.format(variance_score))
        # print("CLF best: {}".format(clf.best_score_)) grid search only
        print('MSE score: {}'.format(MSEscore))
        print('MAE score: {}'.format(MAEscore))
        print('R2 score: {}'.format(R2score))

        # plot for residual error
        plt.style.use('fivethirtyeight')

        # plotting residual errors in training data
        plt.scatter(clf.predict(X_train), clf.predict(X_train) - y_train,
                    color="green", s=10, label='Train data')

        # plotting residual errors in test data
        plt.scatter(clf.predict(X_test), clf.predict(X_test) - y_test,
                    color="blue", s=10, label='Test data')

        # plotting line for zero residual error
        plt.hlines(y=0, xmin=10, xmax=14, linewidth=2)

        plt.legend(loc='upper right')
        plt.title("Residual errors")
        plt.show()
    else:
        return [variance_score, MSEscore, MAEscore, R2score]


if __name__ == "__main__":
    headless_run = False
    run()
