import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Preprocess Data
train = pd.read_csv("datasets_coursework1/real-state/train_full_Real-estate.csv")
train_x = train.drop(columns=["No","Y house price of unit area"])
train_y = train["Y house price of unit area"]
scaler = StandardScaler()
train_x.loc[:,:] = scaler.fit_transform(train_x)

# Fit regression model
model = RandomForestRegressor(n_estimators=100, random_state=2024)
model.fit(train_x, train_y)
# Save model to disk
filename = 'part1_output/real-state-regression.model'
with open(filename, 'wb') as f:
    pickle.dump(dict(model=model, scaler=scaler), f)
print("Regression model saved to", filename)


# Fit classification model
train_y = train_y.apply(lambda x: "Expensive" if x >= 30 else "Not Expensive")
model = RandomForestClassifier(n_estimators=100, random_state=2024)
model.fit(train_x, train_y)
# Save model to disk
filename = 'part1_output/real-state-classification.model'
with open(filename, 'wb') as f:
    pickle.dump(dict(model=model, scaler=scaler), f)
print("Classification model saved to", filename)