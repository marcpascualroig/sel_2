import pandas as pd
from decision_tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random_forest import RandomForest
from decision_forest import DecisionForest

file_path = './Data/mushroom.csv'
data = pd.read_csv(file_path)
#data = pd.read_excel(file_path)

X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values    # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

_, tot_features = X.shape


forest = DecisionForest(tot_features=tot_features, NT=4, F=3, max_depth=2)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(y_test)
print(y_pred)
print("Accuracy:", accuracy)
