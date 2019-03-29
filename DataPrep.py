import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import DataConversionWarning
import warnings


# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def normalize(train):
    # Loop through all nummerical feature columns
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    stdSc = StandardScaler()

    # for column in train.select_dtypes(exclude=["object"]).columns:
        # max = train.loc[:, column].max()
        # min = train.loc[:, column].min()
        # train.loc[:, column] = train.loc[:, column].apply(
        #     lambda x: (x-min)/(max-min))
    train.loc[:, train.select_dtypes(exclude=["object"]).columns] = stdSc.fit_transform(train.loc[:, train.select_dtypes(exclude=["object"]).columns])
    return train


def handle_cat(train):
    # Handle missing values for features where median/mean or most common value doesn't make sense
    # Alley : data description says NA means "no alley access"
    train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("None")
    # BedroomAbvGr : NA most likely means 0
    train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)
    # BsmtQual etc : data description says NA for basement features is "no basement"
    train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
    train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
    train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
    train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
    train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
    train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
    train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
    train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
    # CentralAir : NA most likely means No
    train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
    # Condition : NA most likely means Normal
    train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
    train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")
    # EnclosedPorch : NA most likely means no enclosed porch
    train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)
    # External stuff : NA most likely means average
    train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
    train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")
    # Fence : data description says NA means "no fence"
    train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")
    # FireplaceQu : data description says NA means "no fireplace"
    train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
    train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)
    # Functional : data description says NA means typical
    train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
    # GarageType etc : data description says NA for garage features is "no garage"
    train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
    train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
    train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
    train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
    train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
    train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
    # HalfBath : NA most likely means no half baths above grade
    train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)
    # HeatingQC : NA most likely means typical
    train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")
    # KitchenAbvGr : NA most likely means 0
    train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)
    # KitchenQual : NA most likely means typical
    train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
    # LotFrontage : NA most likely means no lot frontage
    train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
    # LotShape : NA most likely means regular
    train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")
    # MasVnrType : NA most likely means no veneer
    train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
    train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
    # MiscFeature : data description says NA means "no misc feature"
    train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
    train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)
    # OpenPorchSF : NA most likely means no open porch
    train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)
    # PavedDrive : NA most likely means not paved
    train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")
    # PoolQC : data description says NA means "no pool"
    train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
    train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)
    # SaleCondition : NA most likely means normal sale
    train.loc[:, "SaleCondition"] = train.loc[:,
                                              "SaleCondition"].fillna("Normal")
    # ScreenPorch : NA most likely means no screen porch
    train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)
    # TotRmsAbvGrd : NA most likely means 0
    train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)
    # Utilities : NA most likely means all public utilities
    train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")
    # WoodDeckSF : NA most likely means no wood deck
    train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)

    # Some numerical features are actually really categories
    train = train.replace({"MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45",
                                          50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75",
                                          80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120",
                                          150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"},
                           "MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                                      7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
                           })

    # Encode some categorical features as ordered numbers when there is information in the order
    train = train.replace({"Alley": {"Grvl": 1, "Pave": 2},
                           "BsmtCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
                           "BsmtFinType1": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                                            "ALQ": 5, "GLQ": 6},
                           "BsmtFinType2": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                                            "ALQ": 5, "GLQ": 6},
                           "BsmtQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "FireplaceQu": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                                          "Min2": 6, "Min1": 7, "Typ": 8},
                           "GarageCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "GarageQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
                           "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
                           "PavedDrive": {"N": 0, "P": 1, "Y": 2},
                           "PoolQC": {"No": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                           "Street": {"Grvl": 1, "Pave": 2},
                           "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}}
                          )
    return train


def prep_data_deprecated(quiet):
    train = pd.read_csv("data/train.csv")
    # remove outliers of GrLivArea
    train = train[train.GrLivArea < 4000]
    train.SalePrice = np.log1p(train.SalePrice)
    y = train.SalePrice

    train = handle_cat(train)

    # Differentiate numerical features (minus the target) and categorical features
    categorical_features = train.select_dtypes(include=["object"]).columns
    numerical_features = train.select_dtypes(exclude=["object"]).columns
    numerical_features = numerical_features.drop("SalePrice")
    if not quiet:
        print("Numerical features : " + str(len(numerical_features)))
        print("Categorical features : " + str(len(categorical_features)))
    train_num = train[numerical_features]
    train_cat = train[categorical_features]

    # Handle remaining missing values for numerical features by using median as replacement
    if not quiet:
        print("NAs for numerical features in train : " +
          str(train_num.isnull().values.sum()))
    train_num = train_num.fillna(train_num.median())
    if not quiet:
        print("Remaining NAs for numerical features in train : " +
          str(train_num.isnull().values.sum()))

    # Log transform of the skewed numerical features to lessen impact of outliers
    # Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
    # As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
    skewness = train_num.apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    if not quiet:
        print(str(skewness.shape[0]) +
          " skewed numerical features to log transform")
    skewed_features = skewness.index
    train_num[skewed_features] = np.log1p(train_num[skewed_features])

    # Create dummy features for categorical values via one-hot encoding
    if not quiet:
        print("NAs for categorical features in train : " +
          str(train_cat.isnull().values.sum()))
    
    train_cat = pd.get_dummies(train_cat)
    
    if not quiet:
        print("Remaining NAs for categorical features in train : " +
          str(train_cat.isnull().values.sum()))

    # Concatenate numerical and categorical features.
    train = pd.concat([y, train_num, train_cat], axis=1)
    if not quiet:
        print("New number of features : " + str(train.shape[1]))

    return train


# Formerly prep_data_rico
def prep_data(quiet):
    # Data preprocessing
    train = pd.read_csv("./data/train.csv")
    train.SalePrice = np.log1p(train.SalePrice)
    y = train.SalePrice

    # The following lines remove columns with +20% missing values
    # And columns with a high correlation
    train = train.drop(
        ['1stFlrSF', 'GarageYrBlt', 'GarageArea', 'TotRmsAbvGrd'], 1)
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum() / train.isnull().count()
               ).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1,
                             keys=['Total', 'Percent'])
    train = train.drop((missing_data[missing_data['Total'] > 200]).index, 1)

    # Handle missing values for features where median/mean or most common value doesn't make sense
    # BedroomAbvGr : NA most likely means 0
    train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)
    # BsmtQual etc : data description says NA for basement features is "no basement"
    train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
    train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
    train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
    train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
    train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
    train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
    train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
    train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
    # CentralAir : NA most likely means No
    train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
    # Condition : NA most likely means Normal
    train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
    train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")
    # EnclosedPorch : NA most likely means no enclosed porch
    train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)
    # External stuff : NA most likely means average
    train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
    train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")
    train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)
    # Functional : data description says NA means typical
    train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
    # GarageType etc : data description says NA for garage features is "no garage"
    train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
    train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
    train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
    train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
    train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
    # HalfBath : NA most likely means no half baths above grade
    train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)
    # HeatingQC : NA most likely means typical
    train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")
    # KitchenAbvGr : NA most likely means 0
    train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)
    # KitchenQual : NA most likely means typical
    train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
    # LotShape : NA most likely means regular
    train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")
    # MasVnrType : NA most likely means no veneer
    train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
    train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
    train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)
    # OpenPorchSF : NA most likely means no open porch
    train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)
    # PavedDrive : NA most likely means not paved
    train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")
    train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)
    # SaleCondition : NA most likely means normal sale
    train.loc[:, "SaleCondition"] = train.loc[:,
                                              "SaleCondition"].fillna("Normal")
    # ScreenPorch : NA most likely means no screen porch
    train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)
    # Utilities : NA most likely means all public utilities
    train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")
    # WoodDeckSF : NA most likely means no wood deck
    train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)

    # Some numerical features are actually really categories
    train = train.replace({"MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45",
                                          50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75",
                                          80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120",
                                          150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"},
                           "MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                                      7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
                           })

    # Encode some categorical features as ordered numbers when there is information in the order
    train = train.replace({"Alley": {"Grvl": 1, "Pave": 2},
                           "BsmtCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
                           "BsmtFinType1": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                                            "ALQ": 5, "GLQ": 6},
                           "BsmtFinType2": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                                            "ALQ": 5, "GLQ": 6},
                           "BsmtQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "FireplaceQu": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                                          "Min2": 6, "Min1": 7, "Typ": 8},
                           "GarageCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "GarageQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                           "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
                           "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
                           "PavedDrive": {"N": 0, "P": 1, "Y": 2},
                           "PoolQC": {"No": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                           "Street": {"Grvl": 1, "Pave": 2},
                           "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}}
                          )

    # Differentiate numerical features (minus the target) and categorical features
    categorical_features = train.select_dtypes(include=["object"]).columns
    numerical_features = train.select_dtypes(exclude=["object"]).columns
    numerical_features = numerical_features.drop("SalePrice")
    if not quiet:
        print("Numerical features : " + str(len(numerical_features)))
        print("Categorical features : " + str(len(categorical_features)))
    train_num = train[numerical_features]
    train_cat = train[categorical_features]

    # Handle remaining missing values for numerical features by using median as replacement
    if not quiet:
        print("NAs for numerical features in train : " +
          str(train_num.isnull().values.sum()))
    train_num = train_num.fillna(train_num.median())
    
    if not quiet:
        print("Remaining NAs for numerical features in train : " +
          str(train_num.isnull().values.sum()))

    # Log transform of the skewed numerical features to lessen impact of outliers
    # Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
    # As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
    skewness = train_num.apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    if not quiet:
        print(str(skewness.shape[0]) +
          " skewed numerical features to log transform")
    skewed_features = skewness.index
    train_num[skewed_features] = np.log1p(train_num[skewed_features])

    # Create dummy features for categorical values via one-hot encoding
    if not quiet:
        print("NAs for categorical features in train : " +
          str(train_cat.isnull().values.sum()))
    
    train_cat = pd.get_dummies(train_cat)
    
    if not quiet:
        print("Remaining NAs for categorical features in train : " +
          str(train_cat.isnull().values.sum()))

    # Concatenate numerical and categorical features.
    train = pd.concat([y, train_num, train_cat], axis=1)
    if not quiet:
        print("New number of features : " + str(train.shape[1]))

    corrmat = train.corr()
    k = 80  # number of features we keep
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

    return train[cols]
