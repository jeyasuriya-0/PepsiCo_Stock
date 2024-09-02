import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Load your dataset
data = pd.read_csv('PEP.csv')
numeric_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Apply Variance Threshold
variance_threshold = VarianceThreshold(threshold=0.1)
var_thresh_selected = variance_threshold.fit_transform(numeric_data)
selected_features = numeric_data.columns[variance_threshold.get_support()]
print("Selected features by Variance Threshold:", selected_features.tolist())
