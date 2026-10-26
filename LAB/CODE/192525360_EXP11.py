import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
data = {
'Income': [25000, 50000, 75000, 30000, 90000, 40000, 80000, 20000, 65000, 100000],
'Debt': [15000, 10000, 5000, 20000, 3000, 18000, 7000, 22000, 8000, 2000],
'Payment_History': [2, 7, 9, 3, 10, 4, 8, 1, 7, 10],
'Credit_Utilization': [90, 40, 20, 85, 15, 75, 30, 95, 35, 10],
'Credit_Score': ['Poor', 'Standard', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Standard', 'Good']
}
df = pd.DataFrame(data)
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Credit Score Classification")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=le.inverse_transform([0, 1, 2]), zero_division=0))
new_customer = pd.DataFrame([[60000, 9000, 7, 35]], columns=['Income', 'Debt', 'Payment_History', 'Credit_Utilization'])
prediction = model.predict(new_customer)
print("Predicted Credit Score:", le.inverse_transform(prediction)[0])
