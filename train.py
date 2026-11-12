import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Set up the MLflow Experiment
mlflow.set_experiment("Loan_Approval_Experiment")

# Load the data we tracked with DVC
df = pd.read_csv("loan_data_v1.csv")
X = df[['Age', 'Income_Lakhs', 'Credit_Score']]
y = df['Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Start the MLflow Run (The Digital Lab Notebook Page)
with mlflow.start_run():
    
    # Define parameters
    n_trees = 50
    depth = 3
    
    # 3. Log the parameters
    mlflow.log_param("n_estimators", n_trees)
    mlflow.log_param("max_depth", depth)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=n_trees, max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    # 4. Log the metric
    mlflow.log_metric("accuracy", acc)
    
    # 5. Log the actual model artifact (saves the .pkl automatically)
    mlflow.sklearn.log_model(
        model, 
        "random_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )    
    print(f"Run completed. Accuracy: {acc}")