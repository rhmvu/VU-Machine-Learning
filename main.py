import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn import datasets, linear_model, metrics 

# def main():
#     plt.close("all")
#     train = pd.read_csv("./data/train.csv")
#     test = pd.read_csv("./data/test.csv")
#     all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
#                       test.loc[:,'MSSubClass':'SaleCondition']))
#     train.head()
#     matplotlib.rcParams['figure.figsize'] = (24.0, 12.0)
#     # prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
#     # prices.hist()
#     # plt.show()
    
#     train["SalePrice"] = np.log1p(train["SalePrice"])

#     #log transform skewed numeric features:
#     numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

#     skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
#     skewed_feats = skewed_feats[skewed_feats > 0.75]
#     skewed_feats = skewed_feats.index

#     all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

#     all_data = pd.get_dummies(all_data)
#     all_data = all_data.fillna(all_data.mean())

#     X_train = all_data[:train.shape[0]]
#     X_test = all_data[train.shape[0]:]
#     y = train.SalePrice

# def rmse_cv(model):
#     rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
#     return(rmse)

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
all_data = train.loc[:,'MSSubClass':'SaleCondition']
# train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

X_train = all_data[:730]
X_test = all_data[730:]
y_train = train[:730].SalePrice
y_test = train[730:].SalePrice

# create linear regression object 
reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
reg.fit(X_train, y_train) 
  
# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(X_test, y_test))) 
  
# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 
  
## plotting residual errors in training data 
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 
            color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 10, xmax = 14, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 
