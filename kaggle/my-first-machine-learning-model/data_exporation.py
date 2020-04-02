import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
#from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5



#
df_train = pd.read_csv('./train.csv')
#print(df_train.describe())
#print(df_train.head())
#print(df_train.columns)
#
#df_train['SalePrice'] = np.log(df_train['SalePrice'])
#sns.distplot(df_train['SalePrice'], fit=norm)
#plt.figure()
#stats.probplot(df_train['SalePrice'], plot=plt)
#plt.show()

#deleting points
#print(df_train.sort_values(by = 'GrLivArea', ascending = False)[:2].GrLivArea)
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#scatter plot grlivarea/saleprice
#var = 'GrLivArea'
#data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#sns.distplot(df_train['GrLivArea'], fit=norm);
#plt.figure()
#stats.probplot(df_train['GrLivArea'], plot=plt)
#plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
#plt.show()

#scatter plot totalbsmtsf/saleprice
#var = 'TotalBsmtSF'
#data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#sns.distplot(df_train['TotalBsmtSF'], fit=norm)
#plt.figure()
#stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#plt.show()

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
#df_train['HasBsmt'] = 0
#df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
#df_train['TotalBsmtSF'] = np.log(df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'])
#sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
#plt.figure()
#stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#plt.show()

#box plot overallqual/saleprice
#var = 'OverallQual'
#data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#f, ax = plt.subplots(figsize=(8, 6))
#fig = sns.boxplot(x=var, y="SalePrice", data=data)
#fig.axis(ymin=0, ymax=800000);
#plt.show()

#box plot yearbuild/saleprice
#var = 'YearBuilt'
#data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#f, ax = plt.subplots(figsize=(16, 8))
#fig = sns.boxplot(x=var, y="SalePrice", data=data)
#fig.axis(ymin=0, ymax=800000);
#plt.xticks(rotation=90);
#plt.show()

#correlation matrix
#corrmat = df_train.corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
#k = 10 #number of variables for heatmap
#cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
#cm = np.corrcoef(df_train[cols].values.T)
#sns.set(font_scale=1.25)
#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.show()

#scatterplot
#sns.set()
#cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#sns.pairplot(df_train[cols], size = 2.5)
#plt.show()

#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(df_train.PoolQC.isnull().sum())
#print(df_train.PoolQC.isnull().count())
#just checking that there's no missing data missing...
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)




cols = ['OverallQual', 'GrLivArea', 'YearBuilt']
y = df_train.SalePrice
X = df_train[cols]
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25)

# Imputation
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

# Model
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# Add silent=True to avoid printing out updates with each cycle
xgb_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

# Prediction
predictions = xgb_model.predict(test_X)
#predictions = np.exp(predictions)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
print("Score: " + str(rmsle(test_y, predictions)))


# Test
test = pd.read_csv('./test.csv')
#test_X = test.select_dtypes(exclude=['object'])
#test_X = np.log(test[cols])
test_X = test[cols]
test_X = my_imputer.transform(test_X)
predicted_prices = xgb_model.predict(test_X)
#predicted_prices = np.exp(predicted_prices)

print(predicted_prices)

# Submission
#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
#my_submission.to_csv('submission.csv', index=False)
