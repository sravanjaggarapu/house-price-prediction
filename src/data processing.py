import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

house_data = pd.read_csv("house_prices.csv")

house_data.head()

print(house_data["price"])

plt.bar(house_data["bedrooms"],house_data["price"])
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(data=house_data, x= "bedrooms", y= "price")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms VS Price")


plt.figure(figsize=(12,8))
sns.scatterplot(data=house_data, x= "sqft_living", y= "price")
plt.xlabel("squrefeet of living room")
plt.ylabel("Price")
plt.title("Living Room size VS Price")

plt.figure(figsize=(12, 8))
sns.heatmap(house_data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation with Price')
plt.show()