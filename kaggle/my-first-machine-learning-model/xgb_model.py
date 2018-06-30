import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import skew
from math import sqrt
import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

# Load
data = pd.read_csv('./train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

## Log transform skewed numeric features:
#numeric_feats = X.dtypes[X.dtypes != "object"].index
## compute skewness
#skewed_feats = X[numeric_feats].apply(lambda x: skew(x))
#skewed_feats = skewed_feats[skewed_feats > 0.75]
#skewed_feats = skewed_feats.index
#X[skewed_feats] = np.log1p(X[skewed_feats])
#y = np.log1p(y)

train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25)


# Imputation
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)


# Model
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# Add silent=True to avoid printing out updates with each cycle
#xgb_model.fit(train_X, train_y, verbose=False)
xgb_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

# Prediction
predictions = xgb_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
print("Score: " + str(rmsle(test_y, predictions)))
#print("Mean Squared Error : " + str(sqrt(mean_squared_error(test_y, predictions))))

# Test
test = pd.read_csv('./test.csv')
test_X = test.select_dtypes(exclude=['object'])
test_X = my_imputer.transform(test_X)
predicted_prices = xgb_model.predict(test_X)

#predicted_prices = np.expm1(predicted_prices)
print(predicted_prices)

# Submission
#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
#my_submission.to_csv('submission.csv', index=False)
