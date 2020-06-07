# Kaggle Competition: House Prices: Advanced Regression Techniques

Competition Description:

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

https://www.kaggle.com/c/house-prices-advanced-regression-techniques#description

### Table of Contents

<a href='#1. Business Understanding'>1. Business Understanding</a>

<a href='#2. Data Understanding'>2. Data Understanding</a>
Include libraries
Import Data

<a href='#3. Data Preparation'>3. Data Preparation</a>
Handle Missing Values
Feature Engineering (combined features)
Normalization and Scaling

<a href='#4. Modeling'>4. Modeling</a>
Hyperparameter Tuning
Training Models

<a href='#5. Evaluation'>5. Evaluation</a>
Predict Values
Feature Selection
Model Comparison and Selection
Tuning and Ensembling

<a href='#6. Conclusion'>6. Conclusion</a>

<a id='1. Business Understanding'></a>
## 1) Business Understanding

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

<a id='2. Data Understanding'></a>
## 2) Data Understanding

### 2.1 Import Libraries


```python
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import scipy.stats as stats
import math
import os
import sys

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib as mpl
import matplotlib.pylab as pylab
#configure visualizations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6
from pandas.plotting import scatter_matrix

# machine learning
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

# # Modelling Helpers
# from sklearn.preprocessing import Imputer , Normalizer , scale
# from sklearn.cross_validation import train_test_split , StratifiedKFold
# from sklearn.feature_selection import RFECV
```

### 2.2 Load Data


```python
# # save filepath to variable for easier access
# melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

train = pd.read_csv('train.csv')
print(train.shape)
testM = pd.read_csv('test.csv')
print(testM.shape)
#combine = [train, testM]

features = pd.concat([train, testM], keys=['train', 'test'])
```

    (1460, 81)
    (1459, 80)



```python
features_master = pd.concat([train, testM], keys=['train', 'test'])
```


```python
train, test = train_test_split(train, train_size = 0.7)
```

### 2.3.1 Statistical Summaries


```python
train.shape
```




    (1021, 81)




```python
test.shape
```




    (439, 81)




```python
print(train.columns.values)
# Classes?

#train.info()
```

    ['Id' 'MSSubClass' 'MSZoning' 'LotFrontage' 'LotArea' 'Street' 'Alley'
     'LotShape' 'LandContour' 'Utilities' 'LotConfig' 'LandSlope'
     'Neighborhood' 'Condition1' 'Condition2' 'BldgType' 'HouseStyle'
     'OverallQual' 'OverallCond' 'YearBuilt' 'YearRemodAdd' 'RoofStyle'
     'RoofMatl' 'Exterior1st' 'Exterior2nd' 'MasVnrType' 'MasVnrArea'
     'ExterQual' 'ExterCond' 'Foundation' 'BsmtQual' 'BsmtCond' 'BsmtExposure'
     'BsmtFinType1' 'BsmtFinSF1' 'BsmtFinType2' 'BsmtFinSF2' 'BsmtUnfSF'
     'TotalBsmtSF' 'Heating' 'HeatingQC' 'CentralAir' 'Electrical' '1stFlrSF'
     '2ndFlrSF' 'LowQualFinSF' 'GrLivArea' 'BsmtFullBath' 'BsmtHalfBath'
     'FullBath' 'HalfBath' 'BedroomAbvGr' 'KitchenAbvGr' 'KitchenQual'
     'TotRmsAbvGrd' 'Functional' 'Fireplaces' 'FireplaceQu' 'GarageType'
     'GarageYrBlt' 'GarageFinish' 'GarageCars' 'GarageArea' 'GarageQual'
     'GarageCond' 'PavedDrive' 'WoodDeckSF' 'OpenPorchSF' 'EnclosedPorch'
     '3SsnPorch' 'ScreenPorch' 'PoolArea' 'PoolQC' 'Fence' 'MiscFeature'
     'MiscVal' 'MoSold' 'YrSold' 'SaleType' 'SaleCondition' 'SalePrice']



```python
train.dtypes[train.dtypes=='int64']
```




    Id               int64
    MSSubClass       int64
    LotArea          int64
    OverallQual      int64
    OverallCond      int64
    YearBuilt        int64
    YearRemodAdd     int64
    BsmtFinSF1       int64
    BsmtFinSF2       int64
    BsmtUnfSF        int64
    TotalBsmtSF      int64
    1stFlrSF         int64
    2ndFlrSF         int64
    LowQualFinSF     int64
    GrLivArea        int64
    BsmtFullBath     int64
    BsmtHalfBath     int64
    FullBath         int64
    HalfBath         int64
    BedroomAbvGr     int64
    KitchenAbvGr     int64
    TotRmsAbvGrd     int64
    Fireplaces       int64
    GarageCars       int64
    GarageArea       int64
    WoodDeckSF       int64
    OpenPorchSF      int64
    EnclosedPorch    int64
    3SsnPorch        int64
    ScreenPorch      int64
    PoolArea         int64
    MiscVal          int64
    MoSold           int64
    YrSold           int64
    SalePrice        int64
    dtype: object




```python
train.dtypes[train.dtypes=='float']
```




    LotFrontage    float64
    MasVnrArea     float64
    GarageYrBlt    float64
    dtype: object




```python
train.dtypes[train.dtypes=='O']
```




    MSZoning         object
    Street           object
    Alley            object
    LotShape         object
    LandContour      object
    Utilities        object
    LotConfig        object
    LandSlope        object
    Neighborhood     object
    Condition1       object
    Condition2       object
    BldgType         object
    HouseStyle       object
    RoofStyle        object
    RoofMatl         object
    Exterior1st      object
    Exterior2nd      object
    MasVnrType       object
    ExterQual        object
    ExterCond        object
    Foundation       object
    BsmtQual         object
    BsmtCond         object
    BsmtExposure     object
    BsmtFinType1     object
    BsmtFinType2     object
    Heating          object
    HeatingQC        object
    CentralAir       object
    Electrical       object
    KitchenQual      object
    Functional       object
    FireplaceQu      object
    GarageType       object
    GarageFinish     object
    GarageQual       object
    GarageCond       object
    PavedDrive       object
    PoolQC           object
    Fence            object
    MiscFeature      object
    SaleType         object
    SaleCondition    object
    dtype: object




```python
# preview the data
pd.options.display.max_columns=100
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1210</th>
      <td>1211</td>
      <td>60</td>
      <td>RL</td>
      <td>70.0</td>
      <td>11218</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>SawyerW</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>6</td>
      <td>5</td>
      <td>1992</td>
      <td>1992</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>None</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>1055</td>
      <td>1055</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1055</td>
      <td>790</td>
      <td>0</td>
      <td>1845</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1992.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>462</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>635</td>
      <td>104</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>Shed</td>
      <td>400</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>189000</td>
    </tr>
    <tr>
      <th>561</th>
      <td>562</td>
      <td>20</td>
      <td>RL</td>
      <td>77.0</td>
      <td>10010</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>1974</td>
      <td>1975</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>ALQ</td>
      <td>1071</td>
      <td>LwQ</td>
      <td>123</td>
      <td>195</td>
      <td>1389</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1389</td>
      <td>0</td>
      <td>0</td>
      <td>1389</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1975.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>418</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>240</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>170000</td>
    </tr>
    <tr>
      <th>604</th>
      <td>605</td>
      <td>20</td>
      <td>RL</td>
      <td>88.0</td>
      <td>12803</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>2002</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>99.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>922</td>
      <td>Unf</td>
      <td>0</td>
      <td>572</td>
      <td>1494</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1494</td>
      <td>0</td>
      <td>0</td>
      <td>1494</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2002.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>530</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>221000</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>1037</td>
      <td>20</td>
      <td>RL</td>
      <td>89.0</td>
      <td>12898</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Timber</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>9</td>
      <td>5</td>
      <td>2007</td>
      <td>2008</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>Stone</td>
      <td>70.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>TA</td>
      <td>Gd</td>
      <td>GLQ</td>
      <td>1022</td>
      <td>Unf</td>
      <td>0</td>
      <td>598</td>
      <td>1620</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1620</td>
      <td>0</td>
      <td>0</td>
      <td>1620</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Ex</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>Ex</td>
      <td>Attchd</td>
      <td>2008.0</td>
      <td>Fin</td>
      <td>3</td>
      <td>912</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>228</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>315500</td>
    </tr>
    <tr>
      <th>399</th>
      <td>400</td>
      <td>60</td>
      <td>FV</td>
      <td>65.0</td>
      <td>8125</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Somerst</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2006</td>
      <td>2007</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>Stone</td>
      <td>100.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>812</td>
      <td>Unf</td>
      <td>0</td>
      <td>280</td>
      <td>1092</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1112</td>
      <td>438</td>
      <td>0</td>
      <td>1550</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>2007.0</td>
      <td>Fin</td>
      <td>2</td>
      <td>438</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>168</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>10</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>241000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>247</th>
      <td>248</td>
      <td>20</td>
      <td>RL</td>
      <td>75.0</td>
      <td>11310</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>1954</td>
      <td>1954</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>BrkFace</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>1367</td>
      <td>1367</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1375</td>
      <td>0</td>
      <td>0</td>
      <td>1375</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1954.0</td>
      <td>Unf</td>
      <td>2</td>
      <td>451</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>1370</th>
      <td>1371</td>
      <td>50</td>
      <td>RL</td>
      <td>90.0</td>
      <td>5400</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>OldTown</td>
      <td>Artery</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>4</td>
      <td>6</td>
      <td>1920</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CBlock</td>
      <td>CBlock</td>
      <td>None</td>
      <td>0.0</td>
      <td>Fa</td>
      <td>TA</td>
      <td>PConc</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>ALQ</td>
      <td>315</td>
      <td>Rec</td>
      <td>105</td>
      <td>420</td>
      <td>840</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>840</td>
      <td>534</td>
      <td>0</td>
      <td>1374</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Detchd</td>
      <td>1967.0</td>
      <td>Fin</td>
      <td>1</td>
      <td>338</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>198</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>10</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>148</th>
      <td>149</td>
      <td>20</td>
      <td>RL</td>
      <td>63.0</td>
      <td>7500</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>SawyerW</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>2004</td>
      <td>2005</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>120.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>680</td>
      <td>Unf</td>
      <td>0</td>
      <td>400</td>
      <td>1080</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1080</td>
      <td>0</td>
      <td>0</td>
      <td>1080</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>141000</td>
    </tr>
    <tr>
      <th>1128</th>
      <td>1129</td>
      <td>60</td>
      <td>RL</td>
      <td>59.0</td>
      <td>11796</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2004</td>
      <td>2005</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>847</td>
      <td>847</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>847</td>
      <td>1112</td>
      <td>0</td>
      <td>1959</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>BuiltIn</td>
      <td>2004.0</td>
      <td>Fin</td>
      <td>2</td>
      <td>434</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>100</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>853</th>
      <td>854</td>
      <td>80</td>
      <td>RL</td>
      <td>NaN</td>
      <td>12095</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>SLvl</td>
      <td>6</td>
      <td>6</td>
      <td>1964</td>
      <td>1964</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>HdBoard</td>
      <td>BrkFace</td>
      <td>115.0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Rec</td>
      <td>564</td>
      <td>Unf</td>
      <td>0</td>
      <td>563</td>
      <td>1127</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1445</td>
      <td>0</td>
      <td>0</td>
      <td>1445</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Fa</td>
      <td>Attchd</td>
      <td>1964.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>645</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>180</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>158000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>849.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1017.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>961.00000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>739.808031</td>
      <td>56.993144</td>
      <td>70.071849</td>
      <td>10543.216454</td>
      <td>6.100881</td>
      <td>5.576885</td>
      <td>1971.266405</td>
      <td>1984.659158</td>
      <td>99.277286</td>
      <td>440.507346</td>
      <td>49.871694</td>
      <td>560.409403</td>
      <td>1050.788443</td>
      <td>1158.953967</td>
      <td>345.192948</td>
      <td>5.661117</td>
      <td>1509.808031</td>
      <td>0.425073</td>
      <td>0.058766</td>
      <td>1.570029</td>
      <td>0.376102</td>
      <td>2.846229</td>
      <td>1.047013</td>
      <td>6.505387</td>
      <td>0.616063</td>
      <td>1979.16025</td>
      <td>1.759060</td>
      <td>471.482860</td>
      <td>96.534770</td>
      <td>46.208619</td>
      <td>20.795299</td>
      <td>3.396670</td>
      <td>14.307542</td>
      <td>3.945152</td>
      <td>30.568071</td>
      <td>6.285015</td>
      <td>2007.856024</td>
      <td>180939.880509</td>
    </tr>
    <tr>
      <th>std</th>
      <td>420.929544</td>
      <td>41.954912</td>
      <td>24.263797</td>
      <td>10851.770650</td>
      <td>1.413038</td>
      <td>1.125121</td>
      <td>30.077735</td>
      <td>20.929091</td>
      <td>177.145037</td>
      <td>462.753951</td>
      <td>164.625732</td>
      <td>443.306222</td>
      <td>460.268324</td>
      <td>396.897207</td>
      <td>437.267398</td>
      <td>47.474005</td>
      <td>543.166972</td>
      <td>0.515940</td>
      <td>0.239432</td>
      <td>0.553282</td>
      <td>0.500565</td>
      <td>0.763671</td>
      <td>0.220835</td>
      <td>1.606732</td>
      <td>0.650412</td>
      <td>24.02779</td>
      <td>0.755846</td>
      <td>215.061776</td>
      <td>126.169191</td>
      <td>66.495662</td>
      <td>60.386080</td>
      <td>29.826563</td>
      <td>54.769870</td>
      <td>48.002889</td>
      <td>205.671713</td>
      <td>2.656487</td>
      <td>1.353223</td>
      <td>82373.592237</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1910.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>375.000000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7500.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1953.000000</td>
      <td>1966.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>217.000000</td>
      <td>780.000000</td>
      <td>866.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1118.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>1962.00000</td>
      <td>1.000000</td>
      <td>326.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>128500.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>746.000000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9416.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1972.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>381.000000</td>
      <td>0.000000</td>
      <td>480.000000</td>
      <td>980.000000</td>
      <td>1077.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1456.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1981.00000</td>
      <td>2.000000</td>
      <td>478.000000</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>161500.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1107.000000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11584.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>153.000000</td>
      <td>712.000000</td>
      <td>0.000000</td>
      <td>784.000000</td>
      <td>1296.000000</td>
      <td>1391.000000</td>
      <td>727.000000</td>
      <td>0.000000</td>
      <td>1761.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>2002.00000</td>
      <td>2.000000</td>
      <td>576.000000</td>
      <td>169.000000</td>
      <td>66.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>213000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1378.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>2336.000000</td>
      <td>6110.000000</td>
      <td>4692.000000</td>
      <td>2065.000000</td>
      <td>572.000000</td>
      <td>5642.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>3.000000</td>
      <td>2010.00000</td>
      <td>4.000000</td>
      <td>1418.000000</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>3500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 2.3.2 Visualizations

#### Look at outcome variable (SalePrice)


```python
plt.scatter(range(0,len(train['SalePrice'])),train['SalePrice'])
```




    <matplotlib.collections.PathCollection at 0x1285b2c90>




![png](output_24_1.png)



```python
plt.hist(train['SalePrice'])
```




    (array([110., 504., 253.,  94.,  35.,  16.,   3.,   3.,   1.,   2.]),
     array([ 34900., 106910., 178920., 250930., 322940., 394950., 466960.,
            538970., 610980., 682990., 755000.]),
     <a list of 10 Patch objects>)




![png](output_25_1.png)



```python
sale_price_norm = np.log(train['SalePrice'])
plt.hist(sale_price_norm)
```




    (array([  4.,   7.,  39., 139., 326., 275., 148.,  62.,  16.,   5.]),
     array([10.46024211, 10.7676652 , 11.07508829, 11.38251138, 11.68993448,
            11.99735757, 12.30478066, 12.61220375, 12.91962684, 13.22704994,
            13.53447303]),
     <a list of 10 Patch objects>)




![png](output_26_1.png)



```python
print(min(train['SalePrice']))

#sale_price_norm = np.log1p(train['SalePrice'])
#plt.hist(sale_price_norm)

from scipy.stats import norm
#applying log transformation
#train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot
sns.distplot(np.log(train['SalePrice']), fit=norm);
fig = plt.figure()
res = stats.probplot(np.log(train['SalePrice']), plot=plt)
```

    34900



![png](output_27_1.png)



![png](output_27_2.png)


#### Look at Missing Values


```python
#Then we shall be getting the percentage of the missing values in columns of our dataset like below
percentage_missing = train.isnull().sum()/len(train)
percentage_missing = percentage_missing[percentage_missing > 0]
percentage_missing.sort_values(inplace=True)#we use inplace=True to make changes to our columns
print(percentage_missing)


#    # Handle remaining missing values for numerical features by using median as replacement
#    print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
#    train_num = train_num.fillna(train_num.median())
#    print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
```

    MasVnrType      0.003918
    MasVnrArea      0.003918
    BsmtQual        0.030362
    BsmtCond        0.030362
    BsmtExposure    0.030362
    BsmtFinType1    0.030362
    BsmtFinType2    0.031342
    GarageCond      0.058766
    GarageQual      0.058766
    GarageFinish    0.058766
    GarageType      0.058766
    GarageYrBlt     0.058766
    LotFrontage     0.168462
    FireplaceQu     0.473066
    Fence           0.809011
    Alley           0.939275
    MiscFeature     0.962782
    PoolQC          0.993144
    dtype: float64



```python
#lets plot to visualize the missing values
percentage_missing = percentage_missing.to_frame()
percentage_missing.columns=['Count']
percentage_missing.index.names = ['Name']
percentage_missing['Name'] = percentage_missing.index
plt.figure(figsize=(15,15))
sns.barplot(x="Name",y="Count",data=percentage_missing)
plt.xticks(rotation=90)

#missing = train.isnull().sum()
#missing = missing[missing > 0]
#missing.sort_values(inplace=True)
#missing.plot.bar()
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17]), <a list of 18 Text xticklabel objects>)




![png](output_30_1.png)



```python
# Look at the variables with high numbers of missings
for i in list(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']):
    cat_pivot = train.pivot_table(index=i,values="SalePrice",aggfunc=np.median)
    print (cat_pivot)
```

            SalePrice
    PoolQC           
    Ex         745000
    Fa         181000
    Gd         171000
                 SalePrice
    MiscFeature           
    Gar2            170750
    Othr             55000
    Shed            146000
           SalePrice
    Alley           
    Grvl      115000
    Pave      168600
           SalePrice
    Fence           
    GdPrv     170500
    GdWo      137500
    MnPrv     137225
    MnWw      132500
                 SalePrice
    FireplaceQu           
    Ex              311500
    Fa              157500
    Gd              206950
    Po              131500
    TA              186500


We can probably assume that the missings for PoolQC, Alley, Fence, and FireplaceQu correspond with those features being absent in a house.

### 2.3.3 Numeric Data


```python
numVars = list(train.describe().columns)
print(numVars)
#quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
#quantitative.remove('SalePrice')
#quantitative.remove('Id')
#OR
#numerical_data = train.select_dtypes(include=[np.number])
#categorical_data = train.select_dtypes(exclude=[np.number])
#len(quantitative)
print(len(numVars))
numVars.remove('SalePrice')
numVars.remove('Id')
print(len(numVars))
numData = train[numVars]
```

    ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']
    38
    36



```python
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
print(qualitative)
print(len(qualitative))
catData = train[qualitative]
```

    ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
    43


Initial efforts will focus on examining what factors may contribute to graduation rates. Graduate rates help determine whether schools are designated as Title I schools, and thus receive significant funding for special programs to aid students and teachers. 

#### Numeric Data


```python
sns.pairplot(train[numVars[0:6] + list(["SalePrice"])].dropna(), x_vars=numVars[0:6], y_vars='SalePrice', size=2.5, diag_kind="kde")
sns.pairplot(train[numVars[6:12] + list(["SalePrice"])].dropna(), x_vars=numVars[6:12], y_vars='SalePrice', size=2.5, diag_kind="kde")
sns.pairplot(train[numVars[12:18] + list(["SalePrice"])].dropna(), x_vars=numVars[12:18], y_vars='SalePrice', size=2.5, diag_kind="kde")
sns.pairplot(train[numVars[18:24] + list(["SalePrice"])].dropna(), x_vars=numVars[18:24], y_vars='SalePrice', size=2.5, diag_kind="kde")
sns.pairplot(train[numVars[24:30] + list(["SalePrice"])].dropna(), x_vars=numVars[24:30], y_vars='SalePrice', size=2.5, diag_kind="kde")
sns.pairplot(train[numVars[30:36] + list(["SalePrice"])].dropna(), x_vars=numVars[30:36], y_vars='SalePrice', size=2.5, diag_kind="kde")
```




    <seaborn.axisgrid.PairGrid at 0x34a43668>




![png](output_38_1.png)



![png](output_38_2.png)



![png](output_38_3.png)



![png](output_38_4.png)



![png](output_38_5.png)



![png](output_38_6.png)



```python
# Compute the correlation matrix
corr = train[numVars+list(["SalePrice"])].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=plt.cm.PuOr, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MSSubClass</th>
      <td>1.000000</td>
      <td>-0.373626</td>
      <td>-0.103255</td>
      <td>0.005597</td>
      <td>-0.072961</td>
      <td>0.014486</td>
      <td>0.027440</td>
      <td>-0.027402</td>
      <td>-0.048171</td>
      <td>-0.084233</td>
      <td>-0.153056</td>
      <td>-0.233046</td>
      <td>-0.234434</td>
      <td>0.281092</td>
      <td>0.055874</td>
      <td>0.059259</td>
      <td>0.027889</td>
      <td>0.006339</td>
      <td>0.102888</td>
      <td>0.146635</td>
      <td>-0.004496</td>
      <td>0.288701</td>
      <td>0.045535</td>
      <td>-0.036081</td>
      <td>0.063767</td>
      <td>-0.055145</td>
      <td>-0.106732</td>
      <td>0.021815</td>
      <td>0.014135</td>
      <td>-0.029425</td>
      <td>-0.051108</td>
      <td>-0.008266</td>
      <td>0.004490</td>
      <td>-0.006668</td>
      <td>-0.021420</td>
      <td>-0.028886</td>
      <td>-0.101732</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>-0.373626</td>
      <td>1.000000</td>
      <td>0.435467</td>
      <td>0.244137</td>
      <td>-0.031078</td>
      <td>0.121299</td>
      <td>0.095892</td>
      <td>0.210796</td>
      <td>0.243094</td>
      <td>0.054316</td>
      <td>0.128607</td>
      <td>0.395931</td>
      <td>0.464002</td>
      <td>0.093999</td>
      <td>0.019116</td>
      <td>0.419949</td>
      <td>0.088842</td>
      <td>-0.017678</td>
      <td>0.215259</td>
      <td>0.054894</td>
      <td>0.258096</td>
      <td>-0.013936</td>
      <td>0.359546</td>
      <td>0.264978</td>
      <td>0.081562</td>
      <td>0.250700</td>
      <td>0.322623</td>
      <td>0.059977</td>
      <td>0.156827</td>
      <td>-0.054228</td>
      <td>0.083703</td>
      <td>0.054411</td>
      <td>0.216813</td>
      <td>0.008084</td>
      <td>0.008792</td>
      <td>0.010340</td>
      <td>0.342569</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>-0.103255</td>
      <td>0.435467</td>
      <td>1.000000</td>
      <td>0.095322</td>
      <td>0.011183</td>
      <td>0.002927</td>
      <td>-0.000680</td>
      <td>0.088449</td>
      <td>0.204248</td>
      <td>0.139353</td>
      <td>-0.012555</td>
      <td>0.249981</td>
      <td>0.285387</td>
      <td>0.061255</td>
      <td>0.003781</td>
      <td>0.262902</td>
      <td>0.163583</td>
      <td>0.046430</td>
      <td>0.120335</td>
      <td>0.007326</td>
      <td>0.120648</td>
      <td>-0.013794</td>
      <td>0.198045</td>
      <td>0.253788</td>
      <td>-0.029727</td>
      <td>0.134133</td>
      <td>0.155291</td>
      <td>0.145296</td>
      <td>0.066283</td>
      <td>-0.019195</td>
      <td>0.013321</td>
      <td>0.035508</td>
      <td>0.075379</td>
      <td>0.042480</td>
      <td>0.010714</td>
      <td>0.008767</td>
      <td>0.242651</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>0.005597</td>
      <td>0.244137</td>
      <td>0.095322</td>
      <td>1.000000</td>
      <td>-0.080882</td>
      <td>0.579946</td>
      <td>0.569879</td>
      <td>0.402266</td>
      <td>0.230516</td>
      <td>-0.079668</td>
      <td>0.323133</td>
      <td>0.535582</td>
      <td>0.488616</td>
      <td>0.259961</td>
      <td>-0.040030</td>
      <td>0.572183</td>
      <td>0.090976</td>
      <td>-0.048028</td>
      <td>0.552704</td>
      <td>0.249875</td>
      <td>0.095393</td>
      <td>-0.191354</td>
      <td>0.416342</td>
      <td>0.382595</td>
      <td>0.553794</td>
      <td>0.595769</td>
      <td>0.551573</td>
      <td>0.216674</td>
      <td>0.328186</td>
      <td>-0.132430</td>
      <td>0.022643</td>
      <td>0.080288</td>
      <td>0.064370</td>
      <td>-0.032151</td>
      <td>0.098398</td>
      <td>0.002367</td>
      <td>0.787610</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>-0.072961</td>
      <td>-0.031078</td>
      <td>0.011183</td>
      <td>-0.080882</td>
      <td>1.000000</td>
      <td>-0.395940</td>
      <td>0.051409</td>
      <td>-0.134866</td>
      <td>-0.051912</td>
      <td>0.047771</td>
      <td>-0.122473</td>
      <td>-0.159757</td>
      <td>-0.114442</td>
      <td>0.040033</td>
      <td>0.030964</td>
      <td>-0.049876</td>
      <td>-0.075806</td>
      <td>0.130206</td>
      <td>-0.168747</td>
      <td>-0.055563</td>
      <td>0.037948</td>
      <td>-0.091513</td>
      <td>-0.045559</td>
      <td>-0.035004</td>
      <td>-0.342697</td>
      <td>-0.174786</td>
      <td>-0.135555</td>
      <td>0.006698</td>
      <td>-0.015015</td>
      <td>0.084824</td>
      <td>0.004654</td>
      <td>0.088596</td>
      <td>0.018904</td>
      <td>0.062930</td>
      <td>-0.002971</td>
      <td>0.014853</td>
      <td>-0.068277</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>0.014486</td>
      <td>0.121299</td>
      <td>0.002927</td>
      <td>0.579946</td>
      <td>-0.395940</td>
      <td>1.000000</td>
      <td>0.592847</td>
      <td>0.320624</td>
      <td>0.239896</td>
      <td>-0.055678</td>
      <td>0.150064</td>
      <td>0.380864</td>
      <td>0.274969</td>
      <td>-0.007718</td>
      <td>-0.179933</td>
      <td>0.182297</td>
      <td>0.184069</td>
      <td>-0.036021</td>
      <td>0.444453</td>
      <td>0.241653</td>
      <td>-0.062570</td>
      <td>-0.171295</td>
      <td>0.098238</td>
      <td>0.156104</td>
      <td>0.825264</td>
      <td>0.521871</td>
      <td>0.462319</td>
      <td>0.217487</td>
      <td>0.196357</td>
      <td>-0.386972</td>
      <td>0.022938</td>
      <td>-0.042392</td>
      <td>0.025549</td>
      <td>-0.026315</td>
      <td>0.029318</td>
      <td>-0.026893</td>
      <td>0.512285</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>0.027440</td>
      <td>0.095892</td>
      <td>-0.000680</td>
      <td>0.569879</td>
      <td>0.051409</td>
      <td>0.592847</td>
      <td>1.000000</td>
      <td>0.188589</td>
      <td>0.121076</td>
      <td>-0.086140</td>
      <td>0.198794</td>
      <td>0.294721</td>
      <td>0.255816</td>
      <td>0.122864</td>
      <td>-0.039440</td>
      <td>0.287109</td>
      <td>0.099873</td>
      <td>-0.004014</td>
      <td>0.436868</td>
      <td>0.192185</td>
      <td>-0.033202</td>
      <td>-0.140002</td>
      <td>0.202510</td>
      <td>0.131044</td>
      <td>0.637113</td>
      <td>0.437503</td>
      <td>0.376002</td>
      <td>0.206045</td>
      <td>0.243672</td>
      <td>-0.201355</td>
      <td>0.029988</td>
      <td>-0.020707</td>
      <td>0.009886</td>
      <td>-0.012429</td>
      <td>0.054736</td>
      <td>0.028679</td>
      <td>0.519485</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>-0.027402</td>
      <td>0.210796</td>
      <td>0.088449</td>
      <td>0.402266</td>
      <td>-0.134866</td>
      <td>0.320624</td>
      <td>0.188589</td>
      <td>1.000000</td>
      <td>0.250648</td>
      <td>-0.097139</td>
      <td>0.141367</td>
      <td>0.368494</td>
      <td>0.351644</td>
      <td>0.143520</td>
      <td>-0.068155</td>
      <td>0.371533</td>
      <td>0.061808</td>
      <td>0.001257</td>
      <td>0.251400</td>
      <td>0.182491</td>
      <td>0.107154</td>
      <td>-0.041353</td>
      <td>0.303507</td>
      <td>0.253705</td>
      <td>0.262852</td>
      <td>0.346290</td>
      <td>0.374316</td>
      <td>0.133547</td>
      <td>0.136623</td>
      <td>-0.133332</td>
      <td>0.025945</td>
      <td>0.062765</td>
      <td>0.020102</td>
      <td>-0.028907</td>
      <td>-0.009303</td>
      <td>0.001888</td>
      <td>0.434458</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>-0.048171</td>
      <td>0.243094</td>
      <td>0.204248</td>
      <td>0.230516</td>
      <td>-0.051912</td>
      <td>0.239896</td>
      <td>0.121076</td>
      <td>0.250648</td>
      <td>1.000000</td>
      <td>-0.050354</td>
      <td>-0.495309</td>
      <td>0.531483</td>
      <td>0.453131</td>
      <td>-0.138807</td>
      <td>-0.069762</td>
      <td>0.218322</td>
      <td>0.655835</td>
      <td>0.046751</td>
      <td>0.045849</td>
      <td>-0.000520</td>
      <td>-0.112206</td>
      <td>-0.058430</td>
      <td>0.045547</td>
      <td>0.267817</td>
      <td>0.136141</td>
      <td>0.207413</td>
      <td>0.288804</td>
      <td>0.172825</td>
      <td>0.107935</td>
      <td>-0.105451</td>
      <td>0.019881</td>
      <td>0.057661</td>
      <td>0.161643</td>
      <td>0.005609</td>
      <td>-0.011275</td>
      <td>0.029066</td>
      <td>0.361469</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>-0.084233</td>
      <td>0.054316</td>
      <td>0.139353</td>
      <td>-0.079668</td>
      <td>0.047771</td>
      <td>-0.055678</td>
      <td>-0.086140</td>
      <td>-0.097139</td>
      <td>-0.050354</td>
      <td>1.000000</td>
      <td>-0.202925</td>
      <td>0.097809</td>
      <td>0.076269</td>
      <td>-0.125885</td>
      <td>0.028008</td>
      <td>-0.043128</td>
      <td>0.139546</td>
      <td>0.101937</td>
      <td>-0.096592</td>
      <td>-0.065242</td>
      <td>-0.041221</td>
      <td>-0.036911</td>
      <td>-0.067383</td>
      <td>0.010489</td>
      <td>-0.097677</td>
      <td>-0.052342</td>
      <td>-0.030629</td>
      <td>0.078999</td>
      <td>-0.047818</td>
      <td>0.043839</td>
      <td>-0.034563</td>
      <td>0.041678</td>
      <td>0.014987</td>
      <td>-0.002118</td>
      <td>-0.009860</td>
      <td>0.041721</td>
      <td>-0.028977</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>-0.153056</td>
      <td>0.128607</td>
      <td>-0.012555</td>
      <td>0.323133</td>
      <td>-0.122473</td>
      <td>0.150064</td>
      <td>0.198794</td>
      <td>0.141367</td>
      <td>-0.495309</td>
      <td>-0.202925</td>
      <td>1.000000</td>
      <td>0.410585</td>
      <td>0.312389</td>
      <td>0.007582</td>
      <td>0.038283</td>
      <td>0.242450</td>
      <td>-0.440386</td>
      <td>-0.092771</td>
      <td>0.304890</td>
      <td>-0.043954</td>
      <td>0.156032</td>
      <td>0.002762</td>
      <td>0.252892</td>
      <td>0.061309</td>
      <td>0.201892</td>
      <td>0.233980</td>
      <td>0.202333</td>
      <td>-0.004922</td>
      <td>0.147972</td>
      <td>0.022487</td>
      <td>0.019874</td>
      <td>0.006600</td>
      <td>-0.025530</td>
      <td>-0.022735</td>
      <td>0.043569</td>
      <td>-0.035107</td>
      <td>0.227896</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>-0.233046</td>
      <td>0.395931</td>
      <td>0.249981</td>
      <td>0.535582</td>
      <td>-0.159757</td>
      <td>0.380864</td>
      <td>0.294721</td>
      <td>0.368494</td>
      <td>0.531483</td>
      <td>0.097809</td>
      <td>0.410585</td>
      <td>1.000000</td>
      <td>0.812413</td>
      <td>-0.181850</td>
      <td>-0.024697</td>
      <td>0.455110</td>
      <td>0.294030</td>
      <td>-0.007898</td>
      <td>0.318514</td>
      <td>-0.067522</td>
      <td>0.024218</td>
      <td>-0.071299</td>
      <td>0.276541</td>
      <td>0.344680</td>
      <td>0.313753</td>
      <td>0.431996</td>
      <td>0.493033</td>
      <td>0.203475</td>
      <td>0.243735</td>
      <td>-0.072166</td>
      <td>0.028419</td>
      <td>0.081537</td>
      <td>0.148587</td>
      <td>-0.017614</td>
      <td>0.028291</td>
      <td>0.010001</td>
      <td>0.595057</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>-0.234434</td>
      <td>0.464002</td>
      <td>0.285387</td>
      <td>0.488616</td>
      <td>-0.114442</td>
      <td>0.274969</td>
      <td>0.255816</td>
      <td>0.351644</td>
      <td>0.453131</td>
      <td>0.076269</td>
      <td>0.312389</td>
      <td>0.812413</td>
      <td>1.000000</td>
      <td>-0.194319</td>
      <td>-0.009277</td>
      <td>0.586152</td>
      <td>0.230613</td>
      <td>-0.017794</td>
      <td>0.380826</td>
      <td>-0.127571</td>
      <td>0.116856</td>
      <td>0.051045</td>
      <td>0.421253</td>
      <td>0.428095</td>
      <td>0.235899</td>
      <td>0.436928</td>
      <td>0.499005</td>
      <td>0.203169</td>
      <td>0.222954</td>
      <td>-0.041149</td>
      <td>0.032521</td>
      <td>0.089143</td>
      <td>0.159141</td>
      <td>-0.026481</td>
      <td>0.050189</td>
      <td>0.010293</td>
      <td>0.601274</td>
    </tr>
    <tr>
      <th>2ndFlrSF</th>
      <td>0.281092</td>
      <td>0.093999</td>
      <td>0.061255</td>
      <td>0.259961</td>
      <td>0.040033</td>
      <td>-0.007718</td>
      <td>0.122864</td>
      <td>0.143520</td>
      <td>-0.138807</td>
      <td>-0.125885</td>
      <td>0.007582</td>
      <td>-0.181850</td>
      <td>-0.194319</td>
      <td>1.000000</td>
      <td>0.070378</td>
      <td>0.675878</td>
      <td>-0.176407</td>
      <td>-0.032656</td>
      <td>0.417961</td>
      <td>0.603255</td>
      <td>0.511443</td>
      <td>0.065879</td>
      <td>0.616042</td>
      <td>0.196265</td>
      <td>0.048173</td>
      <td>0.166013</td>
      <td>0.117783</td>
      <td>0.104102</td>
      <td>0.237098</td>
      <td>0.031252</td>
      <td>-0.016286</td>
      <td>0.066433</td>
      <td>0.056581</td>
      <td>0.009591</td>
      <td>0.053992</td>
      <td>-0.010684</td>
      <td>0.298172</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>0.055874</td>
      <td>0.019116</td>
      <td>0.003781</td>
      <td>-0.040030</td>
      <td>0.030964</td>
      <td>-0.179933</td>
      <td>-0.039440</td>
      <td>-0.068155</td>
      <td>-0.069762</td>
      <td>0.028008</td>
      <td>0.038283</td>
      <td>-0.024697</td>
      <td>-0.009277</td>
      <td>0.070378</td>
      <td>1.000000</td>
      <td>0.141171</td>
      <td>-0.039118</td>
      <td>-0.027391</td>
      <td>-0.004377</td>
      <td>-0.017023</td>
      <td>0.110433</td>
      <td>0.020034</td>
      <td>0.143973</td>
      <td>-0.013485</td>
      <td>-0.033817</td>
      <td>-0.089636</td>
      <td>-0.073506</td>
      <td>-0.044816</td>
      <td>0.038677</td>
      <td>0.008886</td>
      <td>-0.001225</td>
      <td>0.054732</td>
      <td>-0.008201</td>
      <td>-0.001608</td>
      <td>-0.038767</td>
      <td>-0.020565</td>
      <td>-0.010802</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>0.059259</td>
      <td>0.419949</td>
      <td>0.262902</td>
      <td>0.572183</td>
      <td>-0.049876</td>
      <td>0.182297</td>
      <td>0.287109</td>
      <td>0.371533</td>
      <td>0.218322</td>
      <td>-0.043128</td>
      <td>0.242450</td>
      <td>0.455110</td>
      <td>0.586152</td>
      <td>0.675878</td>
      <td>0.141171</td>
      <td>1.000000</td>
      <td>0.024660</td>
      <td>-0.042338</td>
      <td>0.623751</td>
      <td>0.394617</td>
      <td>0.513577</td>
      <td>0.093504</td>
      <td>0.828643</td>
      <td>0.477623</td>
      <td>0.215608</td>
      <td>0.452658</td>
      <td>0.461110</td>
      <td>0.232109</td>
      <td>0.362732</td>
      <td>-0.004410</td>
      <td>0.010862</td>
      <td>0.125497</td>
      <td>0.163926</td>
      <td>-0.012070</td>
      <td>0.077849</td>
      <td>-0.002896</td>
      <td>0.689907</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>0.027889</td>
      <td>0.088842</td>
      <td>0.163583</td>
      <td>0.090976</td>
      <td>-0.075806</td>
      <td>0.184069</td>
      <td>0.099873</td>
      <td>0.061808</td>
      <td>0.655835</td>
      <td>0.139546</td>
      <td>-0.440386</td>
      <td>0.294030</td>
      <td>0.230613</td>
      <td>-0.176407</td>
      <td>-0.039118</td>
      <td>0.024660</td>
      <td>1.000000</td>
      <td>-0.142109</td>
      <td>-0.103702</td>
      <td>-0.034319</td>
      <td>-0.158916</td>
      <td>-0.040944</td>
      <td>-0.067764</td>
      <td>0.135695</td>
      <td>0.110306</td>
      <td>0.129587</td>
      <td>0.180570</td>
      <td>0.172524</td>
      <td>0.058184</td>
      <td>-0.065470</td>
      <td>-0.001611</td>
      <td>0.000579</td>
      <td>0.064579</td>
      <td>-0.030567</td>
      <td>-0.016748</td>
      <td>0.048860</td>
      <td>0.210625</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>0.006339</td>
      <td>-0.017678</td>
      <td>0.046430</td>
      <td>-0.048028</td>
      <td>0.130206</td>
      <td>-0.036021</td>
      <td>-0.004014</td>
      <td>0.001257</td>
      <td>0.046751</td>
      <td>0.101937</td>
      <td>-0.092771</td>
      <td>-0.007898</td>
      <td>-0.017794</td>
      <td>-0.032656</td>
      <td>-0.027391</td>
      <td>-0.042338</td>
      <td>-0.142109</td>
      <td>1.000000</td>
      <td>-0.061485</td>
      <td>-0.022116</td>
      <td>0.014713</td>
      <td>-0.013208</td>
      <td>-0.023958</td>
      <td>0.019726</td>
      <td>-0.072350</td>
      <td>-0.039175</td>
      <td>-0.033414</td>
      <td>0.035428</td>
      <td>-0.025345</td>
      <td>-0.073701</td>
      <td>0.025927</td>
      <td>0.067038</td>
      <td>-0.016091</td>
      <td>-0.004811</td>
      <td>0.035009</td>
      <td>-0.029937</td>
      <td>-0.046635</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>0.102888</td>
      <td>0.215259</td>
      <td>0.120335</td>
      <td>0.552704</td>
      <td>-0.168747</td>
      <td>0.444453</td>
      <td>0.436868</td>
      <td>0.251400</td>
      <td>0.045849</td>
      <td>-0.096592</td>
      <td>0.304890</td>
      <td>0.318514</td>
      <td>0.380826</td>
      <td>0.417961</td>
      <td>-0.004377</td>
      <td>0.623751</td>
      <td>-0.103702</td>
      <td>-0.061485</td>
      <td>1.000000</td>
      <td>0.106284</td>
      <td>0.379131</td>
      <td>0.122985</td>
      <td>0.571772</td>
      <td>0.273437</td>
      <td>0.453455</td>
      <td>0.469358</td>
      <td>0.401650</td>
      <td>0.173536</td>
      <td>0.292299</td>
      <td>-0.108276</td>
      <td>0.017823</td>
      <td>0.031059</td>
      <td>0.025517</td>
      <td>-0.023660</td>
      <td>0.090328</td>
      <td>-0.000194</td>
      <td>0.559954</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>0.146635</td>
      <td>0.054894</td>
      <td>0.007326</td>
      <td>0.249875</td>
      <td>-0.055563</td>
      <td>0.241653</td>
      <td>0.192185</td>
      <td>0.182491</td>
      <td>-0.000520</td>
      <td>-0.065242</td>
      <td>-0.043954</td>
      <td>-0.067522</td>
      <td>-0.127571</td>
      <td>0.603255</td>
      <td>-0.017023</td>
      <td>0.394617</td>
      <td>-0.034319</td>
      <td>-0.022116</td>
      <td>0.106284</td>
      <td>1.000000</td>
      <td>0.210298</td>
      <td>-0.073496</td>
      <td>0.320747</td>
      <td>0.179981</td>
      <td>0.200628</td>
      <td>0.191946</td>
      <td>0.148052</td>
      <td>0.104218</td>
      <td>0.214175</td>
      <td>-0.103147</td>
      <td>0.009436</td>
      <td>0.060026</td>
      <td>0.027722</td>
      <td>0.010745</td>
      <td>-0.000238</td>
      <td>0.015160</td>
      <td>0.264694</td>
    </tr>
    <tr>
      <th>BedroomAbvGr</th>
      <td>-0.004496</td>
      <td>0.258096</td>
      <td>0.120648</td>
      <td>0.095393</td>
      <td>0.037948</td>
      <td>-0.062570</td>
      <td>-0.033202</td>
      <td>0.107154</td>
      <td>-0.112206</td>
      <td>-0.041221</td>
      <td>0.156032</td>
      <td>0.024218</td>
      <td>0.116856</td>
      <td>0.511443</td>
      <td>0.110433</td>
      <td>0.513577</td>
      <td>-0.158916</td>
      <td>0.014713</td>
      <td>0.379131</td>
      <td>0.210298</td>
      <td>1.000000</td>
      <td>0.187363</td>
      <td>0.669771</td>
      <td>0.112913</td>
      <td>-0.063124</td>
      <td>0.087134</td>
      <td>0.070339</td>
      <td>0.057701</td>
      <td>0.112428</td>
      <td>0.040078</td>
      <td>-0.029468</td>
      <td>0.048631</td>
      <td>0.047681</td>
      <td>-0.003752</td>
      <td>0.079763</td>
      <td>-0.025898</td>
      <td>0.167352</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>0.288701</td>
      <td>-0.013936</td>
      <td>-0.013794</td>
      <td>-0.191354</td>
      <td>-0.091513</td>
      <td>-0.171295</td>
      <td>-0.140002</td>
      <td>-0.041353</td>
      <td>-0.058430</td>
      <td>-0.036911</td>
      <td>0.002762</td>
      <td>-0.071299</td>
      <td>0.051045</td>
      <td>0.065879</td>
      <td>0.020034</td>
      <td>0.093504</td>
      <td>-0.040944</td>
      <td>-0.013208</td>
      <td>0.122985</td>
      <td>-0.073496</td>
      <td>0.187363</td>
      <td>1.000000</td>
      <td>0.251320</td>
      <td>-0.122731</td>
      <td>-0.130878</td>
      <td>-0.049286</td>
      <td>-0.067051</td>
      <td>-0.085576</td>
      <td>-0.087895</td>
      <td>0.040817</td>
      <td>-0.026341</td>
      <td>-0.051267</td>
      <td>-0.015297</td>
      <td>0.074537</td>
      <td>0.023947</td>
      <td>0.042483</td>
      <td>-0.140863</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>0.045535</td>
      <td>0.359546</td>
      <td>0.198045</td>
      <td>0.416342</td>
      <td>-0.045559</td>
      <td>0.098238</td>
      <td>0.202510</td>
      <td>0.303507</td>
      <td>0.045547</td>
      <td>-0.067383</td>
      <td>0.252892</td>
      <td>0.276541</td>
      <td>0.421253</td>
      <td>0.616042</td>
      <td>0.143973</td>
      <td>0.828643</td>
      <td>-0.067764</td>
      <td>-0.023958</td>
      <td>0.571772</td>
      <td>0.320747</td>
      <td>0.669771</td>
      <td>0.251320</td>
      <td>1.000000</td>
      <td>0.343136</td>
      <td>0.149190</td>
      <td>0.365127</td>
      <td>0.342060</td>
      <td>0.160897</td>
      <td>0.248503</td>
      <td>0.004218</td>
      <td>-0.015561</td>
      <td>0.065850</td>
      <td>0.060976</td>
      <td>0.014916</td>
      <td>0.074915</td>
      <td>-0.003224</td>
      <td>0.538350</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>-0.036081</td>
      <td>0.264978</td>
      <td>0.253788</td>
      <td>0.382595</td>
      <td>-0.035004</td>
      <td>0.156104</td>
      <td>0.131044</td>
      <td>0.253705</td>
      <td>0.267817</td>
      <td>0.010489</td>
      <td>0.061309</td>
      <td>0.344680</td>
      <td>0.428095</td>
      <td>0.196265</td>
      <td>-0.013485</td>
      <td>0.477623</td>
      <td>0.135695</td>
      <td>0.019726</td>
      <td>0.273437</td>
      <td>0.179981</td>
      <td>0.112913</td>
      <td>-0.122731</td>
      <td>0.343136</td>
      <td>1.000000</td>
      <td>0.066395</td>
      <td>0.283054</td>
      <td>0.260032</td>
      <td>0.188958</td>
      <td>0.165374</td>
      <td>-0.042423</td>
      <td>-0.002114</td>
      <td>0.151741</td>
      <td>0.094917</td>
      <td>-0.023747</td>
      <td>0.046223</td>
      <td>-0.007528</td>
      <td>0.473254</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>0.063767</td>
      <td>0.081562</td>
      <td>-0.029727</td>
      <td>0.553794</td>
      <td>-0.342697</td>
      <td>0.825264</td>
      <td>0.637113</td>
      <td>0.262852</td>
      <td>0.136141</td>
      <td>-0.097677</td>
      <td>0.201892</td>
      <td>0.313753</td>
      <td>0.235899</td>
      <td>0.048173</td>
      <td>-0.033817</td>
      <td>0.215608</td>
      <td>0.110306</td>
      <td>-0.072350</td>
      <td>0.453455</td>
      <td>0.200628</td>
      <td>-0.063124</td>
      <td>-0.130878</td>
      <td>0.149190</td>
      <td>0.066395</td>
      <td>1.000000</td>
      <td>0.581680</td>
      <td>0.557554</td>
      <td>0.225978</td>
      <td>0.229266</td>
      <td>-0.318399</td>
      <td>0.017758</td>
      <td>-0.064029</td>
      <td>0.011004</td>
      <td>-0.025687</td>
      <td>0.021984</td>
      <td>-0.016226</td>
      <td>0.486775</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>-0.055145</td>
      <td>0.250700</td>
      <td>0.134133</td>
      <td>0.595769</td>
      <td>-0.174786</td>
      <td>0.521871</td>
      <td>0.437503</td>
      <td>0.346290</td>
      <td>0.207413</td>
      <td>-0.052342</td>
      <td>0.233980</td>
      <td>0.431996</td>
      <td>0.436928</td>
      <td>0.166013</td>
      <td>-0.089636</td>
      <td>0.452658</td>
      <td>0.129587</td>
      <td>-0.039175</td>
      <td>0.469358</td>
      <td>0.191946</td>
      <td>0.087134</td>
      <td>-0.049286</td>
      <td>0.365127</td>
      <td>0.283054</td>
      <td>0.581680</td>
      <td>1.000000</td>
      <td>0.876168</td>
      <td>0.228976</td>
      <td>0.220979</td>
      <td>-0.177969</td>
      <td>0.036157</td>
      <td>0.063459</td>
      <td>0.020727</td>
      <td>-0.043165</td>
      <td>0.083575</td>
      <td>-0.025419</td>
      <td>0.638627</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>-0.106732</td>
      <td>0.322623</td>
      <td>0.155291</td>
      <td>0.551573</td>
      <td>-0.135555</td>
      <td>0.462319</td>
      <td>0.376002</td>
      <td>0.374316</td>
      <td>0.288804</td>
      <td>-0.030629</td>
      <td>0.202333</td>
      <td>0.493033</td>
      <td>0.499005</td>
      <td>0.117783</td>
      <td>-0.073506</td>
      <td>0.461110</td>
      <td>0.180570</td>
      <td>-0.033414</td>
      <td>0.401650</td>
      <td>0.148052</td>
      <td>0.070339</td>
      <td>-0.067051</td>
      <td>0.342060</td>
      <td>0.260032</td>
      <td>0.557554</td>
      <td>0.876168</td>
      <td>1.000000</td>
      <td>0.223271</td>
      <td>0.253131</td>
      <td>-0.167784</td>
      <td>0.037626</td>
      <td>0.056710</td>
      <td>0.061110</td>
      <td>-0.026609</td>
      <td>0.066840</td>
      <td>-0.014628</td>
      <td>0.613406</td>
    </tr>
    <tr>
      <th>WoodDeckSF</th>
      <td>0.021815</td>
      <td>0.059977</td>
      <td>0.145296</td>
      <td>0.216674</td>
      <td>0.006698</td>
      <td>0.217487</td>
      <td>0.206045</td>
      <td>0.133547</td>
      <td>0.172825</td>
      <td>0.078999</td>
      <td>-0.004922</td>
      <td>0.203475</td>
      <td>0.203169</td>
      <td>0.104102</td>
      <td>-0.044816</td>
      <td>0.232109</td>
      <td>0.172524</td>
      <td>0.035428</td>
      <td>0.173536</td>
      <td>0.104218</td>
      <td>0.057701</td>
      <td>-0.085576</td>
      <td>0.160897</td>
      <td>0.188958</td>
      <td>0.225978</td>
      <td>0.228976</td>
      <td>0.223271</td>
      <td>1.000000</td>
      <td>0.076420</td>
      <td>-0.126824</td>
      <td>-0.039322</td>
      <td>-0.061449</td>
      <td>0.113725</td>
      <td>-0.011626</td>
      <td>0.037680</td>
      <td>0.037987</td>
      <td>0.288299</td>
    </tr>
    <tr>
      <th>OpenPorchSF</th>
      <td>0.014135</td>
      <td>0.156827</td>
      <td>0.066283</td>
      <td>0.328186</td>
      <td>-0.015015</td>
      <td>0.196357</td>
      <td>0.243672</td>
      <td>0.136623</td>
      <td>0.107935</td>
      <td>-0.047818</td>
      <td>0.147972</td>
      <td>0.243735</td>
      <td>0.222954</td>
      <td>0.237098</td>
      <td>0.038677</td>
      <td>0.362732</td>
      <td>0.058184</td>
      <td>-0.025345</td>
      <td>0.292299</td>
      <td>0.214175</td>
      <td>0.112428</td>
      <td>-0.087895</td>
      <td>0.248503</td>
      <td>0.165374</td>
      <td>0.229266</td>
      <td>0.220979</td>
      <td>0.253131</td>
      <td>0.076420</td>
      <td>1.000000</td>
      <td>-0.102410</td>
      <td>-0.015078</td>
      <td>0.086783</td>
      <td>0.084532</td>
      <td>-0.019991</td>
      <td>0.065408</td>
      <td>-0.057951</td>
      <td>0.330207</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>-0.029425</td>
      <td>-0.054228</td>
      <td>-0.019195</td>
      <td>-0.132430</td>
      <td>0.084824</td>
      <td>-0.386972</td>
      <td>-0.201355</td>
      <td>-0.133332</td>
      <td>-0.105451</td>
      <td>0.043839</td>
      <td>0.022487</td>
      <td>-0.072166</td>
      <td>-0.041149</td>
      <td>0.031252</td>
      <td>0.008886</td>
      <td>-0.004410</td>
      <td>-0.065470</td>
      <td>-0.073701</td>
      <td>-0.108276</td>
      <td>-0.103147</td>
      <td>0.040078</td>
      <td>0.040817</td>
      <td>0.004218</td>
      <td>-0.042423</td>
      <td>-0.318399</td>
      <td>-0.177969</td>
      <td>-0.167784</td>
      <td>-0.126824</td>
      <td>-0.102410</td>
      <td>1.000000</td>
      <td>-0.037236</td>
      <td>-0.081924</td>
      <td>-0.025229</td>
      <td>0.020759</td>
      <td>-0.031269</td>
      <td>0.004313</td>
      <td>-0.132975</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>-0.051108</td>
      <td>0.083703</td>
      <td>0.013321</td>
      <td>0.022643</td>
      <td>0.004654</td>
      <td>0.022938</td>
      <td>0.029988</td>
      <td>0.025945</td>
      <td>0.019881</td>
      <td>-0.034563</td>
      <td>0.019874</td>
      <td>0.028419</td>
      <td>0.032521</td>
      <td>-0.016286</td>
      <td>-0.001225</td>
      <td>0.010862</td>
      <td>-0.001611</td>
      <td>0.025927</td>
      <td>0.017823</td>
      <td>0.009436</td>
      <td>-0.029468</td>
      <td>-0.026341</td>
      <td>-0.015561</td>
      <td>-0.002114</td>
      <td>0.017758</td>
      <td>0.036157</td>
      <td>0.037626</td>
      <td>-0.039322</td>
      <td>-0.015078</td>
      <td>-0.037236</td>
      <td>1.000000</td>
      <td>-0.032019</td>
      <td>-0.008296</td>
      <td>0.001934</td>
      <td>0.022232</td>
      <td>0.021853</td>
      <td>0.029916</td>
    </tr>
    <tr>
      <th>ScreenPorch</th>
      <td>-0.008266</td>
      <td>0.054411</td>
      <td>0.035508</td>
      <td>0.080288</td>
      <td>0.088596</td>
      <td>-0.042392</td>
      <td>-0.020707</td>
      <td>0.062765</td>
      <td>0.057661</td>
      <td>0.041678</td>
      <td>0.006600</td>
      <td>0.081537</td>
      <td>0.089143</td>
      <td>0.066433</td>
      <td>0.054732</td>
      <td>0.125497</td>
      <td>0.000579</td>
      <td>0.067038</td>
      <td>0.031059</td>
      <td>0.060026</td>
      <td>0.048631</td>
      <td>-0.051267</td>
      <td>0.065850</td>
      <td>0.151741</td>
      <td>-0.064029</td>
      <td>0.063459</td>
      <td>0.056710</td>
      <td>-0.061449</td>
      <td>0.086783</td>
      <td>-0.081924</td>
      <td>-0.032019</td>
      <td>1.000000</td>
      <td>-0.018595</td>
      <td>0.006910</td>
      <td>0.032461</td>
      <td>0.040328</td>
      <td>0.146359</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>0.004490</td>
      <td>0.216813</td>
      <td>0.075379</td>
      <td>0.064370</td>
      <td>0.018904</td>
      <td>0.025549</td>
      <td>0.009886</td>
      <td>0.020102</td>
      <td>0.161643</td>
      <td>0.014987</td>
      <td>-0.025530</td>
      <td>0.148587</td>
      <td>0.159141</td>
      <td>0.056581</td>
      <td>-0.008201</td>
      <td>0.163926</td>
      <td>0.064579</td>
      <td>-0.016091</td>
      <td>0.025517</td>
      <td>0.027722</td>
      <td>0.047681</td>
      <td>-0.015297</td>
      <td>0.060976</td>
      <td>0.094917</td>
      <td>0.011004</td>
      <td>0.020727</td>
      <td>0.061110</td>
      <td>0.113725</td>
      <td>0.084532</td>
      <td>-0.025229</td>
      <td>-0.008296</td>
      <td>-0.018595</td>
      <td>1.000000</td>
      <td>-0.005865</td>
      <td>-0.034092</td>
      <td>-0.048339</td>
      <td>0.110834</td>
    </tr>
    <tr>
      <th>MiscVal</th>
      <td>-0.006668</td>
      <td>0.008084</td>
      <td>0.042480</td>
      <td>-0.032151</td>
      <td>0.062930</td>
      <td>-0.026315</td>
      <td>-0.012429</td>
      <td>-0.028907</td>
      <td>0.005609</td>
      <td>-0.002118</td>
      <td>-0.022735</td>
      <td>-0.017614</td>
      <td>-0.026481</td>
      <td>0.009591</td>
      <td>-0.001608</td>
      <td>-0.012070</td>
      <td>-0.030567</td>
      <td>-0.004811</td>
      <td>-0.023660</td>
      <td>0.010745</td>
      <td>-0.003752</td>
      <td>0.074537</td>
      <td>0.014916</td>
      <td>-0.023747</td>
      <td>-0.025687</td>
      <td>-0.043165</td>
      <td>-0.026609</td>
      <td>-0.011626</td>
      <td>-0.019991</td>
      <td>0.020759</td>
      <td>0.001934</td>
      <td>0.006910</td>
      <td>-0.005865</td>
      <td>1.000000</td>
      <td>-0.009926</td>
      <td>-0.001780</td>
      <td>-0.024997</td>
    </tr>
    <tr>
      <th>MoSold</th>
      <td>-0.021420</td>
      <td>0.008792</td>
      <td>0.010714</td>
      <td>0.098398</td>
      <td>-0.002971</td>
      <td>0.029318</td>
      <td>0.054736</td>
      <td>-0.009303</td>
      <td>-0.011275</td>
      <td>-0.009860</td>
      <td>0.043569</td>
      <td>0.028291</td>
      <td>0.050189</td>
      <td>0.053992</td>
      <td>-0.038767</td>
      <td>0.077849</td>
      <td>-0.016748</td>
      <td>0.035009</td>
      <td>0.090328</td>
      <td>-0.000238</td>
      <td>0.079763</td>
      <td>0.023947</td>
      <td>0.074915</td>
      <td>0.046223</td>
      <td>0.021984</td>
      <td>0.083575</td>
      <td>0.066840</td>
      <td>0.037680</td>
      <td>0.065408</td>
      <td>-0.031269</td>
      <td>0.022232</td>
      <td>0.032461</td>
      <td>-0.034092</td>
      <td>-0.009926</td>
      <td>1.000000</td>
      <td>-0.152886</td>
      <td>0.081784</td>
    </tr>
    <tr>
      <th>YrSold</th>
      <td>-0.028886</td>
      <td>0.010340</td>
      <td>0.008767</td>
      <td>0.002367</td>
      <td>0.014853</td>
      <td>-0.026893</td>
      <td>0.028679</td>
      <td>0.001888</td>
      <td>0.029066</td>
      <td>0.041721</td>
      <td>-0.035107</td>
      <td>0.010001</td>
      <td>0.010293</td>
      <td>-0.010684</td>
      <td>-0.020565</td>
      <td>-0.002896</td>
      <td>0.048860</td>
      <td>-0.029937</td>
      <td>-0.000194</td>
      <td>0.015160</td>
      <td>-0.025898</td>
      <td>0.042483</td>
      <td>-0.003224</td>
      <td>-0.007528</td>
      <td>-0.016226</td>
      <td>-0.025419</td>
      <td>-0.014628</td>
      <td>0.037987</td>
      <td>-0.057951</td>
      <td>0.004313</td>
      <td>0.021853</td>
      <td>0.040328</td>
      <td>-0.048339</td>
      <td>-0.001780</td>
      <td>-0.152886</td>
      <td>1.000000</td>
      <td>-0.001440</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>-0.101732</td>
      <td>0.342569</td>
      <td>0.242651</td>
      <td>0.787610</td>
      <td>-0.068277</td>
      <td>0.512285</td>
      <td>0.519485</td>
      <td>0.434458</td>
      <td>0.361469</td>
      <td>-0.028977</td>
      <td>0.227896</td>
      <td>0.595057</td>
      <td>0.601274</td>
      <td>0.298172</td>
      <td>-0.010802</td>
      <td>0.689907</td>
      <td>0.210625</td>
      <td>-0.046635</td>
      <td>0.559954</td>
      <td>0.264694</td>
      <td>0.167352</td>
      <td>-0.140863</td>
      <td>0.538350</td>
      <td>0.473254</td>
      <td>0.486775</td>
      <td>0.638627</td>
      <td>0.613406</td>
      <td>0.288299</td>
      <td>0.330207</td>
      <td>-0.132975</td>
      <td>0.029916</td>
      <td>0.146359</td>
      <td>0.110834</td>
      <td>-0.024997</td>
      <td>0.081784</td>
      <td>-0.001440</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_39_1.png)



```python
#correlation matrix
corrmat = train[numVars+list(['SalePrice'])].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
```


![png](output_40_0.png)



```python
numTop10 = (abs(corr['SalePrice']).sort_values(ascending=False)[1:36])
#print(list(numTop10.index))

sns.barplot(y=list(numTop10.index),  x=numTop10)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x36664940>




![png](output_41_1.png)



```python
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
```


![png](output_42_0.png)



```python
print(abs(corr['SalePrice']).sort_values(ascending=False)[1:10])
numTop10 = (abs(corr['SalePrice']).sort_values(ascending=False)[1:10])
#print(list(numTop10.index))

sns.barplot(y=list(numTop10.index),  x=numTop10)
```

    OverallQual     0.787610
    GrLivArea       0.689907
    GarageCars      0.638627
    GarageArea      0.613406
    1stFlrSF        0.601274
    TotalBsmtSF     0.595057
    FullBath        0.559954
    TotRmsAbvGrd    0.538350
    YearRemodAdd    0.519485
    Name: SalePrice, dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x36d894e0>




![png](output_43_2.png)



```python
corr2 = train[numTop10.index].corr()
#for i in range(1,len(numTop10)+1):
#        print (abs(cm[numTop10.index[i]]).sort_values(ascending=False)[1])

print(corr2>0.6)

for i in range(0,len(corr2)):
    print (corr2[numTop10.index[i]]).sort_values(ascending=False)[1]
```

                  OverallQual  GrLivArea  GarageCars  GarageArea  1stFlrSF  \
    OverallQual          True      False       False       False     False   
    GrLivArea           False       True       False       False     False   
    GarageCars          False      False        True        True     False   
    GarageArea          False      False        True        True     False   
    1stFlrSF            False      False       False       False      True   
    TotalBsmtSF         False      False       False       False      True   
    FullBath            False       True       False       False     False   
    TotRmsAbvGrd        False       True       False       False     False   
    YearRemodAdd        False      False       False       False     False   
    
                  TotalBsmtSF  FullBath  TotRmsAbvGrd  YearRemodAdd  
    OverallQual         False     False         False         False  
    GrLivArea           False      True          True         False  
    GarageCars          False     False         False         False  
    GarageArea          False     False         False         False  
    1stFlrSF             True     False         False         False  
    TotalBsmtSF          True     False         False         False  
    FullBath            False      True         False         False  
    TotRmsAbvGrd        False     False          True         False  
    YearRemodAdd        False     False         False          True  
    0.595768809551
    0.828642621403
    0.876168277443
    0.876168277443
    0.812412546971
    0.812412546971
    0.623750500751
    0.828642621403
    0.5698792141


OverallQual: max corr = 0.59 (include)

GrLivArea: Correlated with FullBath (0.64) and TotRmsAbvGrd (0.83); remove TotRmsAbvGrd

TotalBsmtSF: correlated with 1stFlrSF (0.79); remove 1stFlrSF

GarageCars: correlated with GarageArea (0.89); remove GarageArea

FullBath: max corr = 0.64

YearBuilt: max corr = 0.57


```python
numSelect = list(numTop10.index)
numSelect.remove('TotRmsAbvGrd')
numSelect.remove('1stFlrSF')
numSelect.remove('GarageArea')
```


```python
#scatterplot
sns.set()
#numVars_sub = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'TotalBsmtSF', 'YearBuilt']
#sns.pairplot(train[numVars_sub], size = 2.5)
sns.pairplot(train[list(['SalePrice'])+numSelect], size = 2.5)
plt.show();
```


![png](output_47_0.png)


### 2.3.4 Categorical Data


```python
catData.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSZoning</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinType2</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>KitchenQual</th>
      <th>Functional</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageFinish</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1021</td>
      <td>1021</td>
      <td>57</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1015</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>995</td>
      <td>995</td>
      <td>994</td>
      <td>995</td>
      <td>994</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>1021</td>
      <td>557</td>
      <td>969</td>
      <td>969</td>
      <td>969</td>
      <td>969</td>
      <td>1021</td>
      <td>5</td>
      <td>210</td>
      <td>39</td>
      <td>1021</td>
      <td>1021</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>25</td>
      <td>9</td>
      <td>8</td>
      <td>5</td>
      <td>8</td>
      <td>6</td>
      <td>8</td>
      <td>13</td>
      <td>15</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>6</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>top</th>
      <td>RL</td>
      <td>Pave</td>
      <td>Grvl</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>TA</td>
      <td>Typ</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>Unf</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>Gd</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>810</td>
      <td>1016</td>
      <td>33</td>
      <td>657</td>
      <td>905</td>
      <td>1021</td>
      <td>744</td>
      <td>963</td>
      <td>158</td>
      <td>878</td>
      <td>1007</td>
      <td>849</td>
      <td>520</td>
      <td>800</td>
      <td>1002</td>
      <td>362</td>
      <td>356</td>
      <td>606</td>
      <td>635</td>
      <td>896</td>
      <td>452</td>
      <td>453</td>
      <td>921</td>
      <td>650</td>
      <td>303</td>
      <td>877</td>
      <td>998</td>
      <td>510</td>
      <td>954</td>
      <td>936</td>
      <td>523</td>
      <td>955</td>
      <td>278</td>
      <td>613</td>
      <td>414</td>
      <td>919</td>
      <td>927</td>
      <td>936</td>
      <td>3</td>
      <td>118</td>
      <td>36</td>
      <td>883</td>
      <td>835</td>
    </tr>
  </tbody>
</table>
</div>




```python
for c in list(catData.columns.values):
    train[c] = train[c].astype('category')
    if train[c].isnull().any():
        train[c] = train[c].cat.add_categories(['MISSING'])
        train[c] = train[c].fillna('MISSING')
```


```python
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(train, id_vars=['SalePrice'], value_vars=list(catData.columns.values))
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")
```


![png](output_51_0.png)


Important-looking variables:

Plot variable importance (ANOVA)

Here is quick estimation of influence of categorical variable on SalePrice. For each variable SalePrices are partitioned to distinct sets based on category values. Then check with ANOVA test if sets have similar distributions. If variable has minor impact then set means should be equal. Decreasing pval is sign of increasing diversity in partitions.


```python
def anova(frame):
    anv = pd.DataFrame()
    anv['feature'] = list(catData.columns.values)
    pvals = []
    for c in list(catData.columns.values):
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

a = anova(train)
a['disparity'] = np.log(1./a['pval'].values)
fig, ax = plt.subplots(figsize=(20,17))
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
```

    c:\python27\lib\site-packages\scipy\stats\stats.py:2924: RuntimeWarning: invalid value encountered in double_scalars
      msb = ssbn / float(dfbn)



![png](output_53_1.png)



```python
a_sub = a[a['disparity']>90]
print(a_sub)
```

             feature           pval   disparity
    8   Neighborhood  1.017105e-148  340.765634
    18     ExterQual  2.677525e-143  328.284775
    30   KitchenQual  5.089350e-136  311.524423
    21      BsmtQual  3.937689e-129  295.662883
    34  GarageFinish   2.456914e-73  167.189806
    32   FireplaceQu   8.865292e-72  163.603983
    33    GarageType   2.776186e-62  141.739198
    20    Foundation   4.106332e-61  139.045161
    27     HeatingQC   5.635697e-49  111.097549
    17    MasVnrType   2.189351e-46  105.135309
    24  BsmtFinType1   1.227870e-45  103.411048



```python
cat_sub = train[list(a_sub['feature'])+list(['SalePrice'])]
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(cat_sub, id_vars=['SalePrice'], value_vars=list(a_sub['feature']))
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")
```


![png](output_55_0.png)



```python
quality_vars = list(['ExterQual', 'KitchenQual', 'BsmtQual', 'FireplaceQu', 'HeatingQC'])
for i in quality_vars:
    cat_pivot = train.pivot_table(index=i,values="SalePrice",aggfunc=np.median)
    print (cat_pivot)
```

               SalePrice
    ExterQual           
    Ex            341250
    Fa             81000
    Gd            216837
    TA            139400
                 SalePrice
    KitchenQual           
    Ex              312436
    Fa              115000
    Gd              202500
    TA              137000
              SalePrice
    BsmtQual           
    Ex           314813
    Fa           112000
    Gd           192500
    TA           135500
    MISSING      107750
                 SalePrice
    FireplaceQu           
    Ex              311500
    Fa              157500
    Gd              206950
    Po              131500
    TA              186500
    MISSING         134950
               SalePrice
    HeatingQC           
    Ex            194850
    Fa            120000
    Gd            153250
    TA            135000


<a id='3. Data Preparation'></a>
## 3) Data Preparation

Create a subset of the master data frame (train and test sets) with only the features of interest


```python
featSelect = list(numSelect) + list(cat_sub)
print(featSelect)
```

    ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearRemodAdd', 'Neighborhood', 'ExterQual', 'KitchenQual', 'BsmtQual', 'GarageFinish', 'FireplaceQu', 'GarageType', 'Foundation', 'HeatingQC', 'MasVnrType', 'BsmtFinType1', 'SalePrice']



```python
features = features[featSelect]
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>OverallQual</th>
      <th>GrLivArea</th>
      <th>GarageCars</th>
      <th>TotalBsmtSF</th>
      <th>FullBath</th>
      <th>YearRemodAdd</th>
      <th>Neighborhood</th>
      <th>ExterQual</th>
      <th>KitchenQual</th>
      <th>BsmtQual</th>
      <th>GarageFinish</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>Foundation</th>
      <th>HeatingQC</th>
      <th>MasVnrType</th>
      <th>BsmtFinType1</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">train</th>
      <th>0</th>
      <td>7</td>
      <td>1710</td>
      <td>2.0</td>
      <td>856.0</td>
      <td>2</td>
      <td>2003</td>
      <td>CollgCr</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>RFn</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>BrkFace</td>
      <td>GLQ</td>
      <td>208500.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>1262</td>
      <td>2.0</td>
      <td>1262.0</td>
      <td>2</td>
      <td>1976</td>
      <td>Veenker</td>
      <td>TA</td>
      <td>TA</td>
      <td>Gd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>CBlock</td>
      <td>Ex</td>
      <td>None</td>
      <td>ALQ</td>
      <td>181500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>1786</td>
      <td>2.0</td>
      <td>920.0</td>
      <td>2</td>
      <td>2002</td>
      <td>CollgCr</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>BrkFace</td>
      <td>GLQ</td>
      <td>223500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>1717</td>
      <td>3.0</td>
      <td>756.0</td>
      <td>1</td>
      <td>1970</td>
      <td>Crawfor</td>
      <td>TA</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Unf</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>BrkTil</td>
      <td>Gd</td>
      <td>None</td>
      <td>ALQ</td>
      <td>140000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>2198</td>
      <td>3.0</td>
      <td>1145.0</td>
      <td>2</td>
      <td>2000</td>
      <td>NoRidge</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>BrkFace</td>
      <td>GLQ</td>
      <td>250000.0</td>
    </tr>
  </tbody>
</table>
</div>



### 3.1 Categorical Features

Categorical Features of interest:


```python
print(list(cat_sub.columns.values))
```

    ['Neighborhood', 'ExterQual', 'KitchenQual', 'BsmtQual', 'GarageFinish', 'FireplaceQu', 'GarageType', 'Foundation', 'HeatingQC', 'MasVnrType', 'BsmtFinType1', 'SalePrice']


### Creation of Ordinal Variables (Enumeration)


```python
print(list(quality_vars))
```

    ['ExterQual', 'KitchenQual', 'BsmtQual', 'FireplaceQu', 'HeatingQC']


Replace all Na, NaN values with MISSING category for later processing


```python
#for c in list(catData.columns.values):
for c in list(features.columns.values):
    features[c] = features[c].astype('category')
    if features[c].isnull().any():
        features[c] = features[c].cat.add_categories(['MISSING'])
        features[c] = features[c].fillna('MISSING')
```

(5) Ex - Excellent

(4) Gd - Good

(3) TA - Average/Typical

(2) Fa - Fair

(1) Po - Poor

(NA) Missing


```python
enumList = list()
for i in range(0,len(features)):
    if (features['ExterQual'].iloc[i]=='Ex'):
        enumList.append(5)
    if (features['ExterQual'].iloc[i]=='Gd'):
        enumList.append(4)
    if (features['ExterQual'].iloc[i]=='TA'):
        enumList.append(3)
    if (features['ExterQual'].iloc[i]=='Fa'):
        enumList.append(2)
    if (features['ExterQual'].iloc[i]=='Po'):
        enumList.append(1)
    if (features['ExterQual'].iloc[i]=='MISSING'):
        enumList.append(np.nan)
        
print(np.unique(enumList, return_counts=True))
print(np.unique(features['ExterQual'], return_counts=True))

#train['BsmtQual'][pd.isnull(train['BsmtQual'])]='MISSING'
#train['FireplaceQu'][pd.isnull(train['FireplaceQu'])]='MISSING'

for j in quality_vars:
    enumList = list()
    for i in range(0,len(features)):
        if (features[j].iloc[i]=='Ex'):
            enumList.append(5)
        if (features[j].iloc[i]=='Gd'):
            enumList.append(4)
        if (features[j].iloc[i]=='TA'):
            enumList.append(3)
        if (features[j].iloc[i]=='Fa'):
            enumList.append(2)
        if (features[j].iloc[i]=='Po'):
            enumList.append(1)
        if (features[j].iloc[i]=='MISSING'):
            enumList.append(0)
        if (features[j].iloc[i]=='nan'):
            enumList.append(0)
    features[j+'_E'] = enumList
    
    
#qual_dict={np.nan:0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5}
#name = np.array(['ExterQual','PoolQC' ,'ExterCond','BsmtQual','BsmtCond',\
#                 'HeatingQC','KitchenQual','FireplaceQu', 'GarageQual','GarageCond'])
#for i in name:
#malldata.head()
```

    (array([2, 3, 4, 5]), array([  35, 1798,  979,  107], dtype=int64))
    (array(['Ex', 'Fa', 'Gd', 'TA'], dtype=object), array([ 107,   35,  979, 1798], dtype=int64))



```python
for j in list(['GarageFinish']):
    enumList = list()
    for i in range(0,len(features)):
        if (features[j].iloc[i]=='Fin'):
            enumList.append(3)
        if (features[j].iloc[i]=='RFn'):
            enumList.append(2)
        if (features[j].iloc[i]=='Unf'):
            enumList.append(1)
        if (features[j].iloc[i]=='MISSING'):
            enumList.append(0)
        if (features[j].iloc[i]=='nan'):
            enumList.append(0)
    features[j+'_E'] = enumList
```


```python
quality_vars
```




    ['ExterQual', 'KitchenQual', 'BsmtQual', 'FireplaceQu', 'HeatingQC']




```python
features = features.drop(quality_vars, axis=1)
features = features.drop('GarageFinish', axis=1)
```


```python
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>OverallQual</th>
      <th>GrLivArea</th>
      <th>GarageCars</th>
      <th>TotalBsmtSF</th>
      <th>FullBath</th>
      <th>YearRemodAdd</th>
      <th>Neighborhood</th>
      <th>GarageType</th>
      <th>Foundation</th>
      <th>MasVnrType</th>
      <th>BsmtFinType1</th>
      <th>SalePrice</th>
      <th>ExterQual_E</th>
      <th>KitchenQual_E</th>
      <th>BsmtQual_E</th>
      <th>FireplaceQu_E</th>
      <th>HeatingQC_E</th>
      <th>GarageFinish_E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">train</th>
      <th>0</th>
      <td>7</td>
      <td>1710</td>
      <td>2</td>
      <td>856</td>
      <td>2</td>
      <td>2003</td>
      <td>CollgCr</td>
      <td>Attchd</td>
      <td>PConc</td>
      <td>BrkFace</td>
      <td>GLQ</td>
      <td>208500</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>1262</td>
      <td>2</td>
      <td>1262</td>
      <td>2</td>
      <td>1976</td>
      <td>Veenker</td>
      <td>Attchd</td>
      <td>CBlock</td>
      <td>None</td>
      <td>ALQ</td>
      <td>181500</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>1786</td>
      <td>2</td>
      <td>920</td>
      <td>2</td>
      <td>2002</td>
      <td>CollgCr</td>
      <td>Attchd</td>
      <td>PConc</td>
      <td>BrkFace</td>
      <td>GLQ</td>
      <td>223500</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>1717</td>
      <td>3</td>
      <td>756</td>
      <td>1</td>
      <td>1970</td>
      <td>Crawfor</td>
      <td>Detchd</td>
      <td>BrkTil</td>
      <td>None</td>
      <td>ALQ</td>
      <td>140000</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>2198</td>
      <td>3</td>
      <td>1145</td>
      <td>2</td>
      <td>2000</td>
      <td>NoRidge</td>
      <td>Attchd</td>
      <td>PConc</td>
      <td>BrkFace</td>
      <td>GLQ</td>
      <td>250000</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### Creation of dummy variables


```python
for i in list(['Neighborhood', 'Foundation', 'GarageType', 'BsmtFinType1', 'MasVnrType']):
    cat_pivot = train.pivot_table(index=i,values="SalePrice",aggfunc=np.median)
    print (cat_pivot)
```

                  SalePrice
    Neighborhood           
    Blmngtn          191000
    Blueste          137500
    BrDale           106000
    BrkSide          124300
    ClearCr          200250
    CollgCr          197200
    Crawfor          200624
    Edwards          121750
    Gilbert          181000
    IDOTRR           103000
    MeadowV           88000
    Mitchel          153500
    NAmes            140000
    NPkVill          146000
    NWAmes           182900
    NoRidge          301500
    NridgHt          315000
    OldTown          119000
    SWISU            139500
    Sawyer           135000
    SawyerW          179900
    Somerst          225500
    StoneBr          278000
    Timber           228475
    Veenker          218000
                SalePrice
    Foundation           
    BrkTil         125250
    CBlock         141500
    PConc          205000
    Slab           104150
    Stone          126500
    Wood           164000
                SalePrice
    GarageType           
    2Types         159000
    Attchd         185000
    Basment        148000
    BuiltIn        227500
    CarPort        108000
    Detchd         129500
                  SalePrice
    BsmtFinType1           
    ALQ              149250
    BLQ              139100
    GLQ              213750
    LwQ              139000
    Rec              142000
    Unf              161750
                SalePrice
    MasVnrType           
    BrkCmn         139000
    BrkFace        181000
    None           143000
    Stone          246839


Let's keep Neighborhood, as it is the mostly highly correlated to Sale Price, and turn into dummies


```python
features = features.drop('Foundation', axis=1)
features = features.drop('GarageType', axis=1)
features = features.drop('BsmtFinType1', axis=1)
features = features.drop('MasVnrType', axis=1)
```


```python
one_hot = pd.get_dummies(features['Neighborhood'])
features = features.drop('Neighborhood', axis=1)
features = features.join(one_hot)
print(features.shape)
features.describe()
```

    (2919, 42)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ExterQual_E</th>
      <th>KitchenQual_E</th>
      <th>BsmtQual_E</th>
      <th>FireplaceQu_E</th>
      <th>HeatingQC_E</th>
      <th>GarageFinish_E</th>
      <th>Blmngtn</th>
      <th>Blueste</th>
      <th>BrDale</th>
      <th>BrkSide</th>
      <th>ClearCr</th>
      <th>CollgCr</th>
      <th>Crawfor</th>
      <th>Edwards</th>
      <th>Gilbert</th>
      <th>IDOTRR</th>
      <th>MeadowV</th>
      <th>Mitchel</th>
      <th>NAmes</th>
      <th>NPkVill</th>
      <th>NWAmes</th>
      <th>NoRidge</th>
      <th>NridgHt</th>
      <th>OldTown</th>
      <th>SWISU</th>
      <th>Sawyer</th>
      <th>SawyerW</th>
      <th>Somerst</th>
      <th>StoneBr</th>
      <th>Timber</th>
      <th>Veenker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
      <td>2919.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.396711</td>
      <td>3.509764</td>
      <td>3.477561</td>
      <td>1.768071</td>
      <td>4.151764</td>
      <td>1.715999</td>
      <td>0.009592</td>
      <td>0.003426</td>
      <td>0.010277</td>
      <td>0.036999</td>
      <td>0.015074</td>
      <td>0.091470</td>
      <td>0.035286</td>
      <td>0.066461</td>
      <td>0.056526</td>
      <td>0.031860</td>
      <td>0.012676</td>
      <td>0.039054</td>
      <td>0.151764</td>
      <td>0.007879</td>
      <td>0.044878</td>
      <td>0.024323</td>
      <td>0.056869</td>
      <td>0.081877</td>
      <td>0.016444</td>
      <td>0.051730</td>
      <td>0.042823</td>
      <td>0.062350</td>
      <td>0.017472</td>
      <td>0.024666</td>
      <td>0.008222</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.580293</td>
      <td>0.665273</td>
      <td>0.905448</td>
      <td>1.806619</td>
      <td>0.957952</td>
      <td>0.897327</td>
      <td>0.097486</td>
      <td>0.058440</td>
      <td>0.100873</td>
      <td>0.188792</td>
      <td>0.121867</td>
      <td>0.288325</td>
      <td>0.184534</td>
      <td>0.249129</td>
      <td>0.230975</td>
      <td>0.175658</td>
      <td>0.111889</td>
      <td>0.193758</td>
      <td>0.358854</td>
      <td>0.088431</td>
      <td>0.207072</td>
      <td>0.154078</td>
      <td>0.231631</td>
      <td>0.274225</td>
      <td>0.127197</td>
      <td>0.221519</td>
      <td>0.202492</td>
      <td>0.241832</td>
      <td>0.131043</td>
      <td>0.155132</td>
      <td>0.090317</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(features.shape)
features.head()
```

    (2918, 38)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>OverallQual</th>
      <th>GrLivArea</th>
      <th>GarageCars</th>
      <th>TotalBsmtSF</th>
      <th>FullBath</th>
      <th>YearRemodAdd</th>
      <th>SalePrice</th>
      <th>ExterQual_E</th>
      <th>KitchenQual_E</th>
      <th>BsmtQual_E</th>
      <th>FireplaceQu_E</th>
      <th>HeatingQC_E</th>
      <th>GarageFinish_E</th>
      <th>Blmngtn</th>
      <th>Blueste</th>
      <th>BrDale</th>
      <th>BrkSide</th>
      <th>ClearCr</th>
      <th>CollgCr</th>
      <th>Crawfor</th>
      <th>Edwards</th>
      <th>Gilbert</th>
      <th>IDOTRR</th>
      <th>MeadowV</th>
      <th>Mitchel</th>
      <th>NAmes</th>
      <th>NPkVill</th>
      <th>NWAmes</th>
      <th>NoRidge</th>
      <th>NridgHt</th>
      <th>OldTown</th>
      <th>SWISU</th>
      <th>Sawyer</th>
      <th>SawyerW</th>
      <th>Somerst</th>
      <th>StoneBr</th>
      <th>Timber</th>
      <th>Veenker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">train</th>
      <th>0</th>
      <td>7</td>
      <td>1710</td>
      <td>2</td>
      <td>856</td>
      <td>2</td>
      <td>2003</td>
      <td>208500</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>1262</td>
      <td>2</td>
      <td>1262</td>
      <td>2</td>
      <td>1976</td>
      <td>181500</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>1786</td>
      <td>2</td>
      <td>920</td>
      <td>2</td>
      <td>2002</td>
      <td>223500</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>1717</td>
      <td>3</td>
      <td>756</td>
      <td>1</td>
      <td>1970</td>
      <td>140000</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>2198</td>
      <td>3</td>
      <td>1145</td>
      <td>2</td>
      <td>2000</td>
      <td>250000</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### MISSINGS / IMPUTATION


```python

```


```python
print("Missings or NAs in numeric variables:")
print(" ")
num_missing = features.isnull().sum()
print(num_missing)

print(" ")
print("Missings or NAs in categoric variables:")
print(" ")
for i in list(['ExterQual_E', 'KitchenQual_E', 'BsmtQual_E', 'GarageFinish_E']):
    print(i + ": ")
    print((features[i]==0).sum())
```

    Missings or NAs in numeric variables:
     
    OverallQual       0
    GrLivArea         0
    GarageCars        0
    TotalBsmtSF       0
    FullBath          0
    YearRemodAdd      0
    SalePrice         0
    ExterQual_E       0
    KitchenQual_E     0
    BsmtQual_E        0
    FireplaceQu_E     0
    HeatingQC_E       0
    GarageFinish_E    0
    Blmngtn           0
    Blueste           0
    BrDale            0
    BrkSide           0
    ClearCr           0
    CollgCr           0
    Crawfor           0
    Edwards           0
    Gilbert           0
    IDOTRR            0
    MeadowV           0
    Mitchel           0
    NAmes             0
    NPkVill           0
    NWAmes            0
    NoRidge           0
    NridgHt           0
    OldTown           0
    SWISU             0
    Sawyer            0
    SawyerW           0
    Somerst           0
    StoneBr           0
    Timber            0
    Veenker           0
    dtype: int64
     
    Missings or NAs in categoric variables:
     
    ExterQual_E: 
    0
    KitchenQual_E: 
    0
    BsmtQual_E: 
    81
    GarageFinish_E: 
    159


Upon closer inspection, we see that the missing values (now 0's) only occur in Kitchen Quality, Basement Quality, and Garage Finish variables. Are these truely missing, or could it be that there are just no basements and garages for these homes?

Let's start by removing the single row that has a missing Kitchen Quality variable - likely insignificant


```python
print(features.shape)
features = features.drop(features[features['KitchenQual_E'] == 0].index)
print(features.shape)
```

    (2918, 42)
    (2918, 42)



```python
bsmtqual_0 = features['TotalBsmtSF'][features['BsmtQual_E']==0]
print(bsmtqual_0['train'])
print(bsmtqual_0['test'])
```

    17      0
    39      0
    90      0
    102     0
    156     0
    182     0
    259     0
    342     0
    362     0
    371     0
    392     0
    520     0
    532     0
    533     0
    553     0
    646     0
    705     0
    736     0
    749     0
    778     0
    868     0
    894     0
    897     0
    984     0
    1000    0
    1011    0
    1035    0
    1045    0
    1048    0
    1049    0
    1090    0
    1179    0
    1216    0
    1218    0
    1232    0
    1321    0
    1412    0
    Name: TotalBsmtSF, dtype: category
    Categories (1059, object): [0, 105, 160, 173, ..., 3206, 5095, 6110, MISSING]
    125           0
    133           0
    269           0
    318           0
    354           0
    387           0
    388           0
    396           0
    397           0
    398           0
    400           0
    455           0
    590           0
    606           0
    608           0
    660     MISSING
    662           0
    728           0
    729           0
    730           0
    733           0
    756           0
    757         173
    758         356
    764           0
    927           0
    975           0
    992           0
    993           0
    1030          0
    1038          0
    1087          0
    1092          0
    1104          0
    1118          0
    1139          0
    1242          0
    1303          0
    1306          0
    1343          0
    1344          0
    1364          0
    1431          0
    1444          0
    Name: TotalBsmtSF, dtype: category
    Categories (1059, object): [0, 105, 160, 173, ..., 3206, 5095, 6110, MISSING]



```python
bsmtqual_0 = train['GarageArea'][train['GarageFinish_E']==0]
print(bsmtqual_0['train'])
print(bsmtqual_0['test'])
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-211-fc05b08b3cf1> in <module>()
    ----> 1 bsmtqual_0 = train['GarageArea'][train['GarageFinish_E']==0]
          2 print(bsmtqual_0['train'])
          3 print(bsmtqual_0['test'])


    c:\python27\lib\site-packages\pandas\core\frame.pyc in __getitem__(self, key)
       2137             return self._getitem_multilevel(key)
       2138         else:
    -> 2139             return self._getitem_column(key)
       2140 
       2141     def _getitem_column(self, key):


    c:\python27\lib\site-packages\pandas\core\frame.pyc in _getitem_column(self, key)
       2144         # get column
       2145         if self.columns.is_unique:
    -> 2146             return self._get_item_cache(key)
       2147 
       2148         # duplicate columns & possible reduce dimensionality


    c:\python27\lib\site-packages\pandas\core\generic.pyc in _get_item_cache(self, item)
       1840         res = cache.get(item)
       1841         if res is None:
    -> 1842             values = self._data.get(item)
       1843             res = self._box_item_values(item, values)
       1844             cache[item] = res


    c:\python27\lib\site-packages\pandas\core\internals.pyc in get(self, item, fastpath)
       3836 
       3837             if not isna(item):
    -> 3838                 loc = self.items.get_loc(item)
       3839             else:
       3840                 indexer = np.arange(len(self.items))[isna(self.items)]


    c:\python27\lib\site-packages\pandas\core\indexes\base.pyc in get_loc(self, key, method, tolerance)
       2522                 return self._engine.get_loc(key)
       2523             except KeyError:
    -> 2524                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2525 
       2526         indexer = self.get_indexer([key], method=method, tolerance=tolerance)


    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'GarageFinish_E'



```python

```


```python

```


```python

```

No basements in these homes!


```python
train['GarageFinish_E'][train['GarageArea']==0]
```




    148     0
    1218    0
    434     0
    843     0
    710     0
    750     0
    108     0
    375     0
    464     0
    784     0
    441     0
    163     0
    613     0
    1449    0
    738     0
    165     0
    1257    0
    705     0
    976     0
    99      0
    954     0
    1337    0
    1030    0
    528     0
    562     0
    921     0
    1123    0
    1137    0
    1325    0
    1219    0
    614     0
    88      0
    1326    0
    582     0
    287     0
    1323    0
    1283    0
    535     0
    968     0
    386     0
    127     0
    636     0
    1173    0
    495     0
    649     0
    1407    0
    39      0
    638     0
    960     0
    1143    0
    210     0
    250     0
    1009    0
    198     0
    291     0
    970     0
    533     0
    1011    0
    125     0
    1096    0
    Name: GarageFinish_E, dtype: int64



No garages in these homes!

Let's keep both of these missings as 0's as they line up with the trends we see in Sale Price

#### OUTLIERS

No obvious outliers seen in numeric variables pairplots above. Otherwise, would remove individual rows containing outliers.

#### NORMALIZATION OF SKEWED


```python
## Standardizing numeric features
numeric_features = features.loc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]
numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()

ax = sns.pairplot(numeric_features_standardized)
```


```python
log
```


```python

```


```python
#    from scipy.stats import skew 
#    skewness = train_num.apply(lambda x: skew(x))
#   skewness.sort_values(ascending=False)

#    skewness = skewness[abs(skewness)>0.5]
#    skewness.index

#    skew_features = train[skewness.index]
#    skew_features.columns

#    #we can treat skewness of a feature with the help fof log transformation.so we'll apply the same here.
#    skew_features = np.log1p(skew_features)
```


```python

```


```python
np.log(train['SalePrice'])


```


```python

```


```python
f = pd.melt(train, id_vars=['SalePrice'], value_vars=list(['ExterQual_E', 'KitchenQual_E', 'BsmtQual_E', 'FireplaceQu_E', 'HeatingQC_E', 'GarageFinish_E']))
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")
```


![png](output_105_0.png)



```python
#correlation matrix
corrmat = train[list(['ExterQual_E', 'KitchenQual_E', 'BsmtQual_E', 'FireplaceQu_E', 'HeatingQC_E', 'GarageFinish_E'])+list(['SalePrice'])].corr()
f, ax = plt.subplots(figsize=(8, 5))
hm = sns.heatmap(corrmat, vmax=.8, square=True, annot=True);
plt.show()
```


![png](output_106_0.png)



```python
#corr2 = train[numTop10.index].corr()
print(corrmat>0.6)
```

                    ExterQual_E  KitchenQual_E  BsmtQual_E  FireplaceQu_E  \
    ExterQual_E            True           True       False          False   
    KitchenQual_E          True           True       False          False   
    BsmtQual_E            False          False        True          False   
    FireplaceQu_E         False          False       False           True   
    HeatingQC_E           False          False       False          False   
    GarageFinish_E        False          False       False          False   
    SalePrice              True           True       False          False   
    
                    HeatingQC_E  GarageFinish_E  SalePrice  
    ExterQual_E           False           False       True  
    KitchenQual_E         False           False       True  
    BsmtQual_E            False           False      False  
    FireplaceQu_E         False           False      False  
    HeatingQC_E            True           False      False  
    GarageFinish_E        False            True      False  
    SalePrice             False           False       True  


Keep External, Kitchen and Basement quality metrics. Discard Fireplace and Heating quality.


```python
varFinal = numSelect + list(['ExterQual_E', 'KitchenQual_E', 'BsmtQual_E', 'GarageFinish_E']) + list(['Neighborhood'])
trainF = train[varFinal + list(['SalePrice'])]
trainF.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OverallQual</th>
      <th>GrLivArea</th>
      <th>TotalBsmtSF</th>
      <th>GarageCars</th>
      <th>FullBath</th>
      <th>YearBuilt</th>
      <th>ExterQual_E</th>
      <th>KitchenQual_E</th>
      <th>BsmtQual_E</th>
      <th>GarageFinish_E</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.088149</td>
      <td>1500.682664</td>
      <td>1041.706170</td>
      <td>1.766895</td>
      <td>1.564153</td>
      <td>1971.630754</td>
      <td>3.399608</td>
      <td>3.515181</td>
      <td>3.479922</td>
      <td>1.728697</td>
      <td>180316.541626</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.378429</td>
      <td>515.014060</td>
      <td>407.993546</td>
      <td>0.749196</td>
      <td>0.552222</td>
      <td>29.664427</td>
      <td>0.567899</td>
      <td>0.657546</td>
      <td>0.874944</td>
      <td>0.903072</td>
      <td>79629.881143</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1875.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.000000</td>
      <td>1128.000000</td>
      <td>784.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1954.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>130000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
      <td>1442.000000</td>
      <td>983.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1972.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>1756.000000</td>
      <td>1286.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2000.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>212900.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.000000</td>
      <td>4476.000000</td>
      <td>3200.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>2010.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
</div>



<a id='4. Modeling'></a> 
## 4) MODELING


```python
import copy
X_train_rare = copy.copy(X_train)
X_test_rare = copy.copy(X_test)
X_train_rare["test"]=0
X_test_rare["test"]=1
temp_df = pandas.concat([X_train_rare,X_test_rare],axis=0)
names = list(X_train_rare.columns.values)
temp_df = pandas.concat([X_train_rare,X_test_rare],axis=0)
for i in names:
    temp_df.loc[temp_df[i].value_counts()[temp_df[i]].values < 20, i] = "RARE_VALUE"
for i in range(temp_df.shape[1]):
    temp_df.iloc[:,i]=temp_df.iloc[:,i].astype('str')
X_train_rare = temp_df[temp_df["test"]=="0"].iloc[:,:-1].values
X_test_rare = temp_df[temp_df["test"]=="1"].iloc[:,:-1].values
for i in range(X_train_rare.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(temp_df.iloc[:,:-1].iloc[:, i])
    les.append(le)
    X_train_rare[:, i] = le.transform(X_train_rare[:, i])
    X_test_rare[:, i] = le.transform(X_test_rare[:, i])
enc.fit(X_train_rare)
X_train_rare = enc.transform(X_train_rare)
X_test_rare = enc.transform(X_test_rare)
l.fit(X_train_rare,y_train)
y_pred = l.predict_proba(X_test_rare)
print(log_loss(y_test,y_pred))
r.fit(X_train_rare,y_train)
y_pred = r.predict_proba(X_test_rare)
print(log_loss(y_test,y_pred))
print(X_train_rare.shape)
```


```python

```


```python

```

XGBoost, Ridge, Lasso and Elastic-Net regularization
https://www.kaggle.com/tannercarbonati/detailed-data-analysis-ensemble-modeling

### Lasso Model


```python
def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))

def log_transform(feature):
    train[feature] = np.log1p(train[feature].values)

def quadratic(feature):
    train[feature+'2'] = train[feature]**2
    
log_transform('GrLivArea')
log_transform('1stFlrSF')
log_transform('2ndFlrSF')
log_transform('TotalBsmtSF')
log_transform('LotArea')
log_transform('LotFrontage')
log_transform('KitchenAbvGr')
log_transform('GarageArea')

quadratic('OverallQual')
quadratic('YearBuilt')
quadratic('YearRemodAdd')
quadratic('TotalBsmtSF')
quadratic('2ndFlrSF')
quadratic('Neighborhood_E')
quadratic('RoofMatl_E')
quadratic('GrLivArea')

qdr = ['OverallQual2', 'YearBuilt2', 'YearRemodAdd2', 'TotalBsmtSF2',
        '2ndFlrSF2', 'Neighborhood_E2', 'RoofMatl_E2', 'GrLivArea2']

train['HasBasement'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasGarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train['Has2ndFloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasMasVnr'] = train['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
train['HasWoodDeck'] = train['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasPorch'] = train['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasPool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train['IsNew'] = train['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

boolean = ['HasBasement', 'HasGarage', 'Has2ndFloor', 'HasMasVnr', 'HasWoodDeck',
            'HasPorch', 'HasPool', 'IsNew']


features = quantitative + qual_encoded + boolean + qdr
lasso = linear_model.LassoLarsCV(max_iter=10000)
X = train[features].fillna(0.).values
Y = train['SalePrice'].values
lasso.fit(X, np.log(Y))

Ypred = np.exp(lasso.predict(X))
error(Y, Ypred)
```

Can also use lasso or ridge regression as wrapper filter methods for best feature selection


```python
import patsy

Y, X = patsy.dmatrices(
    "SalePrice ~ \
        GarageCars + \
        np.log1p(BsmtFinSF1) + \
        ScreenPorch + \
        Condition1_E + \
        Condition2_E + \
        WoodDeckSF + \
        np.log1p(LotArea) + \
        Foundation_E + \
        MSZoning_E + \
        MasVnrType_E + \
        HouseStyle_E + \
        Fireplaces + \
        CentralAir_E + \
        BsmtFullBath + \
        EnclosedPorch + \
        PavedDrive_E + \
        ExterQual_E + \
        bs(OverallCond, df=7, degree=1) + \
        bs(MSSubClass, df=7, degree=1) + \
        bs(LotArea, df=2, degree=1) + \
        bs(FullBath, df=3, degree=1) + \
        bs(HalfBath, df=2, degree=1) + \
        bs(BsmtFullBath, df=3, degree=1) + \
        bs(TotRmsAbvGrd, df=2, degree=1) + \
        bs(LandSlope_E, df=2, degree=1) + \
        bs(LotConfig_E, df=2, degree=1) + \
        bs(SaleCondition_E, df=3, degree=1) + \
        OverallQual + np.square(OverallQual) + \
        GrLivArea + np.square(GrLivArea) + \
        Q('1stFlrSF') + np.square(Q('1stFlrSF')) + \
        Q('2ndFlrSF') + np.square(Q('2ndFlrSF')) +  \
        TotalBsmtSF + np.square(TotalBsmtSF) +  \
        KitchenAbvGr + np.square(KitchenAbvGr) +  \
        YearBuilt + np.square(YearBuilt) + \
        Neighborhood_E + np.square(Neighborhood_E) + \
        Neighborhood_E:OverallQual + \
        MSSubClass:BldgType_E + \
        ExterQual_E:OverallQual + \
        PoolArea:PoolQC_E + \
        Fireplaces:FireplaceQu_E + \
        OverallQual:KitchenQual_E + \
        GarageQual_E:GarageCond + \
        GarageArea:GarageCars + \
        Q('1stFlrSF'):TotalBsmtSF + \
        TotRmsAbvGrd:GrLivArea",
    train.to_dict('list'))

ridge = linear_model.RidgeCV(cv=10)
ridge.fit(X, np.log(Y))
Ypred = np.exp(ridge.predict(X))
print(error(Y,Ypred))
```


```python
# Prints R2 and RMSE scores
def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))

# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)
```
