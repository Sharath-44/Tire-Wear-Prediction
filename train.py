import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('final.csv')

X = data.drop('remaining_distance_km', axis=1)
y = data['remaining_distance_km']

categorical_cols = ['thread_pattern', 'road_conditions', 'driving_style',
                    'maintenance_history', 'driving_environment', 'tire_composition', 'tire_size']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

model = xgb.XGBRegressor()
model.fit(X, y)

model.save_model('trained_model.model')

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

plt.scatter(y, y_pred)
plt.xlabel('Actual Remaining Distance (km)')
plt.ylabel('Predicted Remaining Distance (km)')
plt.title('Actual vs Predicted Remaining Distance')
plt.show()
