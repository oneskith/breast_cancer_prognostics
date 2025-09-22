from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.config import *
import pandas as pd

#load data
def train_model():
    df = pd.read_csv(PROCESSED_DATA / "cleaned_data.csv")

    #Define vars
    X = df[df.columns.drop("diagnosis")]
    y = df["diagnosis"]


    #modelling
    print("---Generating model----")
    pipeline = Pipeline([
        ("Scaler", StandardScaler()),
        ("log_reg", LogisticRegression(max_iter=500))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()



