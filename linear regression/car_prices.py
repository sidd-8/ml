import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("used_car_dataset.csv")

# Simple example: numerical features only
features = ['year', 'mileage', 'engineSize']
X = data[features].values
y = data['price'].values

# We scale features for gradient descent: Z-Score Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)