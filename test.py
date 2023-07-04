import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

loaded_model = xgb.Booster()
loaded_model.load_model('trained_model.model')

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

new_data = pd.read_csv('test_wo.csv')

categorical_cols = ['thread_pattern', 'road_conditions', 'driving_style',
                    'maintenance_history', 'driving_environment', 'tire_composition', 'tire_size']

for col in categorical_cols:
    if col in label_encoders:
        le = label_encoders[col]
        new_data[col] = new_data[col].map(
            lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        new_data[col] = le.transform(new_data[col])

data_dmatrix = xgb.DMatrix(new_data)
predictions = loaded_model.predict(data_dmatrix)

true_values = pd.read_csv('test.csv')['remaining_distance_km']

plt.plot(predictions, label='Predicted')
plt.plot(true_values, label='True')
plt.xlabel('Data Instance')
plt.ylabel('Remaining Distance (km)')
plt.title('Predicted vs True Remaining Distance')
plt.legend()
plt.show()

mse = mean_squared_error(true_values, predictions)
mae = mean_absolute_error(true_values, predictions)
r2 = r2_score(true_values, predictions)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)
