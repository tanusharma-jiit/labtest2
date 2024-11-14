import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head(10))
print(df.describe())
df.fillna(df.mean() , inplace=True)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = np.log(y_train)
y_test = np.log(y_test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
ridge_model = Ridge()
param_grid = {'alpha': np.logspace(-3, 3, 7)}
grid_search = GridSearchCV(ridge_model, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)
best_ridge = grid_search.best_estimator_
print("Optimal alpha:", grid_search.best_params_['alpha'])
print("R^2 :", r2_score(y_test, best_ridge.predict(X_test)))
print("MSE :", mean_squared_error(y_test, best_ridge.predict(X_test)))
y_pred = linear_model.predict(X_test)
print("R^2 (Linear):", r2_score(y_test, y_pred))
print("MAE (Linear):", mean_absolute_error(y_test, y_pred))
print("RMSE (Linear):", np.sqrt(mean_squared_error(y_test, y_pred)))
def fun(arr):
    reversed_arr = []
    sorted_strings = ["".join(sorted(x)) for x in arr]
    for i, s in enumerate(arr):
        if sorted_strings.count(sorted_strings[i]) > 1:
            reversed_arr.append(s[::-1])
        else:  
            reversed_arr.append(s)
    return np.array(reversed_arr)

arr = np.array(["listen", "isilent", "enlisit", "googlee", "gooegl"])
print(fun(arr))
