import csv
import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew

from sklearn.linear_model import Ridge, LassoCV
from sklearn.model_selection import cross_val_score

# Somewhat more advanced regression, performs regularization to eliminate some variables that are not necessary

plt.close("all")
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
train.head()
matplotlib.rcParams['figure.figsize'] = (24.0, 12.0)

train["SalePrice"] = np.log1p(train["SalePrice"])

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) # compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


# Root mean square error
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


model_ridge = Ridge()

# Testing the right alpha value for our model
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean()
            for alpha in alphas]


# Trying L1 regularization
model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean()

# Lasso gives us an alpha of 0.1231, picks some coefficients and gives the rest a 0 value
coef = pd.Series(model_lasso.coef_, index=X_train.columns)

# Plotting Residuals
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true": y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x="preds", y="residuals", kind="scatter")
print(plt.show())
