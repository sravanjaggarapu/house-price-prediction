from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#linear regression model and evaluation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import plot_tree

#XGBoost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor




house_data = pd.read_csv('house_prices.csv')

x = house_data[["bedrooms"]]
y = house_data[["price"]]

# LinearRegression implementation
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)
linear_regressor = LinearRegression()
linear_regressor.fit(x_train,y_train)
y_pred = linear_regressor.predict(x_test)

print(y_pred)

np.sqrt(mean_squared_error(y_test,y_pred))


# Decision Tree implementation

tree_regressor = DecisionTreeRegressor()
tree_regressor.fit(x_train,y_train)
y_pred = tree_regressor.predict(x_test)

mean_squared_error(y_pred,y_test)

r2_score(y_pred,y_test)

plt.figure(figsize=(16, 12))
plot_tree(tree_regressor, feature_names=x.columns, filled=True, fontsize=10)
plt.title("Decision Tree for House Price Prediction")
plt.show()

#XGBoost implementation

X = house_data[["bedrooms", "sqft_living", "bathrooms"]]
y = house_data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)