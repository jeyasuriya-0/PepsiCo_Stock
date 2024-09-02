import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv('PEP.csv')
numeric_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Prepare the data
X = numeric_data
y = (data['Close'].shift(-1) > data['Close']).astype(int)
X = X[:-1]  # Drop the last row due to NaN in the target
y = y[:-1]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply RFE
model = LogisticRegression(solver='liblinear')
rfe = RFE(model, n_features_to_select=2)
rfe.fit(X_scaled, y)
rfe_selected_features = X.columns[rfe.support_]
print("Selected features by RFE:", rfe_selected_features.tolist())
