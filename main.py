import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import skew

from sklearn import linear_model

# Constants
TRAIN_DATA_PROPORTION = 0.7

# Very basic linear regression model, takes into account all variables

# Data preprocessing
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
all_data = train.loc[:, 'MSSubClass':'SaleCondition']

# log transform skewed numeric features:
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

X_test = all_data[treshold:1460]
y_test = train[treshold:1460].SalePrice


# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: \n', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(round(reg.score(X_test, y_test), 3)))


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