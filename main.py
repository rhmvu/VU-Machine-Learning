import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

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

X_train = all_data[:730]
X_test = all_data[730:]
y_train = train[:730].SalePrice
y_test = train[730:].SalePrice
print(train['SalePrice'].describe())

#Heatmap
#corrmat = otherTrain.corr()
#f, ax = plt.subplots(figsize=(15, 12))
#sns.heatmap(corrmat, vmax=.8, square=True);
#plt.show();

#The following lines remove columns with +20% missing values
#And columns with a high correlation
#print(train)
train = train.drop(['1stFlrSF', 'GarageYrBlt', 'GarageArea', 'GrLivArea'], 1)
#print(train)
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
df_train = train.drop((missing_data[missing_data['Total'] > 200]).index,1)


#fill the empty spots in the features
le = preprocessing.LabelEncoder()
print(df_train.columns.values)
onlyNumerical = pd.get_dummies(df_train)
#le.fit(df_train)
#imp = Imputer(missing_values=np.nan, strategy='most_frequent')
#filledData = imp.fit(df_train)
#filledData = imp.transform(filledData)
print(onlyNumerical)

exit()
# create linear regression object 
reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
reg.fit(X_train, y_train) 
  
# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(X_test, y_test)))
  
# plot for residual error 
  
# setting plot style
plt.style.use('fivethirtyeight') 
  
# plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 
            color = "green", s = 10, label = 'Train data') 
  
# plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 
            color = "blue", s = 10, label = 'Test data') 
  
# plotting line for zero residual error
plt.hlines(y = 0, xmin = 10, xmax = 14, linewidth = 2) 
  
# plotting legend
plt.legend(loc = 'upper right') 
  
# plot title
plt.title("Residual errors") 
  
# function to show plot
print(plt.show())



