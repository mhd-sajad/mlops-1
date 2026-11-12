import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Connect to the exact same experiment name you used before
mlflow.set_experiment("Loan_Approval_Experiment")

# Load the DVC-tracked dataset
df = pd.read_csv("loan_data_v1.csv")
X = df[['Age', 'Income_Lakhs', 'Credit_Score']]
y = df['Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Start a new digital lab notebook page (Run)
with mlflow.start_run():
    
    # Define and log parameters for this specific model
    c_value = 1.0
    mlflow.log_param("algorithm", "Logistic Regression")
    mlflow.log_param("C_value", c_value)
    
    # Train the new model
    model = LogisticRegression(C=c_value, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    # Log the metric
    mlflow.log_metric("accuracy", acc)
    
    # Log the model artifact (Logistic Regression doesn't use Trees, so no skops exception needed)
    mlflow.sklearn.log_model(model, "logistic_regression_model")
    
    print(f"Logistic Regression run completed. Accuracy: {acc}")