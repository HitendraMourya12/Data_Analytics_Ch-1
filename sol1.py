import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("smartphone_price_dataset.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())

df = df.drop_duplicates()

df = pd.get_dummies(df, columns=['Brand'], drop_first=True)

print("\nCleaned Dataset Shape:", df.shape)



plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()



plt.figure()
sns.scatterplot(x=df['RAM_GB'], y=df['Price_INR'])
plt.title("RAM vs Smartphone Price")
plt.xlabel("RAM (GB)")
plt.ylabel("Price (INR)")
plt.show()

plt.figure()
sns.scatterplot(x=df['Storage_GB'], y=df['Price_INR'])
plt.title("Storage vs Smartphone Price")
plt.xlabel("Storage (GB)")
plt.ylabel("Price (INR)")
plt.show()

plt.figure()
sns.scatterplot(x=df['Battery_mAh'], y=df['Price_INR'])
plt.title("Battery Capacity vs Smartphone Price")
plt.xlabel("Battery Capacity (mAh)")
plt.ylabel("Price (INR)")
plt.show()

plt.figure()
sns.histplot(df['Price_INR'], bins=15, kde=True)
plt.title("Distribution of Smartphone Prices")
plt.show()



X = df.drop("Price_INR", axis=1)
y = df["Price_INR"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))



coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Importance:")
print(coefficients.sort_values(by="Coefficient", ascending=False))
