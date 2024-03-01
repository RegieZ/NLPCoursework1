import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier

# Preprocess Data
test = pd.read_csv("datasets_coursework1/real-state/test_full_Real-estate.csv")
test_x = test.drop(columns=["No","Y house price of unit area"])
test_y = test["Y house price of unit area"]

# Regression model
filename = 'part1_output/real-state-regression.model'
with open(filename, 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']
test_x.loc[:,:] = scaler.transform(test_x)
pred_y = model.predict(test_x)
rmse = np.sqrt(metrics.mean_squared_error(test_y, pred_y))
print(f"Regression RMSE: {rmse:.2f}")

# Classification model
test_y = test_y.apply(lambda x: "Expensive" if x >= 30 else "Not Expensive")
filename = 'part1_output/real-state-classification.model'
with open(filename, 'rb') as f:
    data = pickle.load(f)
    model = data['model']
pred_y = model.predict(test_x)
accuracy = metrics.accuracy_score(test_y, pred_y)
print(f"Classification Accuracy: {accuracy:.2%}")
