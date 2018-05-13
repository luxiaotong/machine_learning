import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


data_file_path = './train.csv'
data = pd.read_csv(data_file_path)
#print(data.describe())

y = data.SalePrice
predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
                        'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[predictors]
#print(X.describe())

# Predict
iowa_model = DecisionTreeRegressor()
iowa_model.fit(X, y)
predictions = iowa_model.predict([[10000, 2000, 1000, 1000, 2, 3, 8]])

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are:")
print(predictions)

# MAE
predicted_home_prices = iowa_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))

# Split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 3)
model = DecisionTreeRegressor()
model.fit(train_X, train_y)
# Validation
val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# max_leaf_nodes
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# Random Forest
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
iowa_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, iowa_preds))

# Test
test = pd.read_csv('./test.csv')
test_X = test[predictors]
predicted_prices = forest_model.predict(test_X)
print(predicted_prices)

# Submission
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)
