import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("titanic (2).csv")

# Keep only required columns
df = df[["Pclass","Sex","Age","Fare","Survived"]]

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Convert Male/Female into numbers
df["Sex"] = df["Sex"].map({"male":0,"female":1})

# Features
X = df.drop("Survived",axis=1)

# Target
y = df["Survived"]

# Split dataset
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Create SVM model
model = SVC(kernel="linear")

# Train model
model.fit(X_train,y_train)

# Prediction
y_pred=model.predict(X_test)

# Accuracy
accuracy=accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)

# Save model
joblib.dump(model,"svm_model.pkl")

print("Model Saved Successfully")