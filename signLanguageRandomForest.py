import pandas as pd

print("Loading data...")
dataset = pd.read_csv("archive/BSL-leap-motion.csv")
print("Data loaded.")

X = dataset.iloc[:,:-1] # Data
y = dataset.iloc[:,-1] # Class labels

print("Splitting into training and testing data...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training and testing datasets created.")


print("\nTraining model...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
model = RandomForestClassifier(n_estimators=50)
print(model)

import time
start = time.time()

model.fit(X_train,y_train)

end = time.time()

print("Model trained in: " + str(end - start) + " seconds")

print("\nTesting model...")

pred_values = model.predict(X_test)

acc = accuracy_score(y_test , pred_values)
prec = precision_score(y_test , pred_values, average="weighted")
rec = recall_score(y_test , pred_values, average="weighted")
f1 = f1_score(y_test , pred_values, average="weighted")
report = classification_report(y_test , pred_values)

print("\nValidation metrics:")
print("Accuracy: " + str(acc*100) + "%")
print("Precision: " + str(prec))
print("Recall: " + str(rec))
print("F1-Score: " + str(f1))

print("\nClassification Report:")
print(report)
