import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('final.csv')

# 1. Bar Plot
plt.figure(figsize=(10, 6))
sns.countplot(x='thread_pattern', data=df)
plt.title('Thread Pattern Distribution')
plt.xlabel('Thread Pattern')
plt.ylabel('Count')
plt.show()

# 2. Histogram
plt.figure(figsize=(10, 6))
sns.histplot(x='temperature_celsius', data=df, bins=10)
plt.title('Temperature Distribution')
plt.xlabel('Temperature (Celsius)')
plt.ylabel('Count')
plt.show()

# 3. Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tire_pressure_psi', y='remaining_distance_km', data=df)
plt.title('Tire Pressure vs Remaining Distance')
plt.xlabel('Tire Pressure (PSI)')
plt.ylabel('Remaining Distance (km)')
plt.show()

# 4. Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='road_conditions', y='mileage_km', data=df)
plt.title('Mileage Distribution across Road Conditions')
plt.xlabel('Road Conditions')
plt.ylabel('Mileage (km)')
plt.show()
