import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'PEP.csv'
data = pd.read_csv(file_path)

# Creating the target variable (1 if Close > Open, else 0)
data['Target'] = (data['Close'] > data['Open']).astype(int)

# Define features and target
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
logreg = LogisticRegression()
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Step 1: Baseline Performance (No Feature Selection)
logreg_baseline_acc, logreg_baseline_report = evaluate_model(logreg, X_train_scaled, X_test_scaled, y_train, y_test)
rf_baseline_acc, rf_baseline_report = evaluate_model(rf_clf, X_train, X_test, y_train, y_test)

# Step 2: Variance Threshold Feature Selection
var_thresh = VarianceThreshold(threshold=0.01)
X_train_vt = var_thresh.fit_transform(X_train_scaled)
X_test_vt = var_thresh.transform(X_test_scaled)

logreg_vt_acc, logreg_vt_report = evaluate_model(logreg, X_train_vt, X_test_vt, y_train, y_test)
rf_vt_acc, rf_vt_report = evaluate_model(rf_clf, X_train_vt, X_test_vt, y_train, y_test)

# Step 3: Recursive Feature Elimination (RFE) Feature Selection
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=3)
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)

logreg_rfe_acc, logreg_rfe_report = evaluate_model(logreg, X_train_rfe, X_test_rfe, y_train, y_test)
rf_rfe_acc, rf_rfe_report = evaluate_model(rf_clf, X_train_rfe, X_test_rfe, y_train, y_test)

# Compile the results for comparison
results = {
    "Baseline": {
        "Logistic Regression": {"Accuracy": logreg_baseline_acc, "Report": logreg_baseline_report},
        "Random Forest": {"Accuracy": rf_baseline_acc, "Report": rf_baseline_report}
    },
    "Variance Threshold": {
        "Logistic Regression": {"Accuracy": logreg_vt_acc, "Report": logreg_vt_report},
        "Random Forest": {"Accuracy": rf_vt_acc, "Report": rf_vt_report}
    },
    "RFE": {
        "Logistic Regression": {"Accuracy": logreg_rfe_acc, "Report": logreg_rfe_report},
        "Random Forest": {"Accuracy": rf_rfe_acc, "Report": rf_rfe_report}
    }
}

# Display the results
for method, result in results.items():
    print(f"\n{method} Feature Selection:")
    for model_name, metrics in result.items():
        print(f"\n{model_name}:")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print("Classification Report:\n", metrics['Report'])
