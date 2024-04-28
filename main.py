import pandas as pd
from decision_tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_path = './Data/ContactLens.csv'
data = pd.read_csv(file_path)

X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values    # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X
X_test = X
y_train = y
y_test = y


# Train the decision tree
print("decision tree")
tree = DecisionTreeClassifier(X_train, max_depth=None)
frequencies = tree.fit(X_train, y_train)
print("end decision tree")
# Make predictions
y_pred = tree.predict(X_test)
print(y_pred)
print(y_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
y_pred = tree.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print("Accuracy:", accuracy)