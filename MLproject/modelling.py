import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Credit_CI")

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "loan_preprocessing/loan_clean.csv"
    df = pd.read_csv(data_path)
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # Autolog akan log metrik dan model

if __name__ == "__main__":
    main()