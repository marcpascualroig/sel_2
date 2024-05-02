import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random_forest import RandomForest
from decision_forest import DecisionForest
import matplotlib.pyplot as plt


#parameters
test_size=0.3
NT = [1, 10, 25, 50, 75, 100]
random_forest = False
decision_forest= False

#datasets = ["ENB2012_data"]
datasets = ["ContactLens"]
datasets = ["mushroom"]
datasets = ["Dry_Bean_Dataset"]
#datasets = ["wdbc"]

for dataset in datasets:
    #data = pd.read_csv("./Data/" + dataset + ".data")
    data = pd.read_excel("./Data/" + dataset + ".xlsx")

    X = data.iloc[:, :-1].values  # Features
    num_instances, tot_features = X.shape
    print('n:', num_instances, "M:", tot_features)

    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    column_names = data.columns

    num_instances, tot_features = X.shape
    forest = RandomForest(tot_features=tot_features, NT=1, F=1, max_depth=None)
    freqs, _ = forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    print(y_pred)
    print(y_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print('freqs', freqs)

    # Create a bar plot
    plt.figure(figsize=(8, 6))  # Set the figure size (width, height) in inches
    plt.bar(column_names[:-1], freqs, color='skyblue')
    # Add labels and title
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Plot of Categories')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # Save the plot to a file (e.g., PNG format)
    plt.savefig('bar_plot.png')


for dataset in datasets:

    print(dataset)
    data = pd.read_csv("./Data/" + dataset + ".csv")
    #data = pd.read_excel("./Data/" + dataset + ".xlsx")
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values
    print(X)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    num_instances, tot_features = X.shape

    print('n:', num_instances, "M:", tot_features)

    #RANDOM FOREST
    if random_forest:
        accuracy_dic = {}
        freqs_dic = {}
        freqs_2_dic = {}
        print("RANDOM FOREST")
        F = [1, 2, int(math.log2(tot_features)) + 1, int(math.sqrt(tot_features))]
        for num_trees in NT:
            print("number of trees:", num_trees)
            for num_features in F:
                print("number of features:", num_features)
                forest = RandomForest(tot_features=tot_features, NT=num_trees, F=num_features, max_depth=None)
                freqs, freqs2 = forest.fit(X_train, y_train)
                y_pred = forest.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print("Accuracy:", accuracy)
                print(freqs)

                accuracy_dic[str(num_trees) + "-" + str(num_features)] = accuracy
                freqs_dic[str(num_trees) + "-" + str(num_features)] = freqs
                freqs_2_dic[str(num_trees) + "-" + str(num_features)] = freqs2

        with open(dataset + "_random_forest.txt", 'w') as file:
            file.write("ACCURACY\n")
            for key, value in accuracy_dic.items():
                file.write(f"{key}: {value}\n")
            file.write("FREQUENCIES\n")
            for key, value in freqs_dic.items():
                file.write(f"{key}: {value}\n")
            file.write("FREQUENCIES 2\n")
            for key, value in freqs_2_dic.items():
                file.write(f"{key}: {value}\n")

        max_key = max(accuracy_dic, key=accuracy_dic.get)
        best_freqs = freqs_dic[max_key]
        best_freqs_2 = freqs_2_dic[max_key]

        column_names = data.columns
        # Create a bar plot
        plt.figure(figsize=(8, 6))  # Set the figure size (width, height) in inches
        plt.bar(column_names[:-1], best_freqs, color='skyblue')
        # Add labels and title
        plt.xlabel('Features')
        plt.ylabel('Frequency')
        plt.title('Frequency of features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Save the plot to a file (e.g., PNG format)
        plt.savefig(r'.\Plots\random_forest_' + dataset + 'frequencies.png')

        plt.figure(figsize=(8, 6))  # Set the figure size (width, height) in inches
        plt.bar(column_names[:-1], best_freqs_2, color='skyblue')
        # Add labels and title
        plt.xlabel('Features')
        plt.ylabel('Frequency')
        plt.title('Frequency of features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Save the plot to a file (e.g., PNG format)
        plt.savefig(r'.\Plots\random_forest_' + dataset + 'frequencies_2.png')

    #DECISION FOREST
    if decision_forest:
        accuracy_dic = {}
        freqs_dic = {}
        freqs_2_dic = {}
        print("DECISION FOREST")
        F = [int(tot_features/4), int(tot_features/2), int(3*tot_features/4), "random"]
        for num_trees in NT:
            print("number of trees:", num_trees)
            for num_features in F:
                print("number of features:", num_features)
                forest = DecisionForest(tot_features=tot_features, NT=num_trees, F=num_features, max_depth=None)
                freqs, freqs2 = forest.fit(X_train, y_train)
                y_pred = forest.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print("Accuracy:", accuracy)
                print(freqs)

                accuracy_dic[str(num_trees) + "-" + str(num_features)] = accuracy
                freqs_dic[str(num_trees) + "-" + str(num_features)] = freqs
                freqs_2_dic[str(num_trees) + "-" + str(num_features)] = freqs2

        with open(dataset + "_decision_forest.txt", 'w') as file:
            file.write("ACCURACY\n")
            for key, value in accuracy_dic.items():
                file.write(f"{key}: {value}\n")
            file.write("FREQUENCIES\n")
            for key, value in freqs_dic.items():
                file.write(f"{key}: {value}\n")
            file.write("FREQUENCIES 2\n")
            for key, value in freqs_2_dic.items():
                file.write(f"{key}: {value}\n")


        max_key = max(accuracy_dic, key=accuracy_dic.get)
        best_freqs = freqs_dic[max_key]
        best_freqs_2 = freqs_2_dic[max_key]

        column_names = data.columns
        # Create a bar plot
        plt.figure(figsize=(8, 6))  # Set the figure size (width, height) in inches
        plt.bar(column_names[:-1], best_freqs, color='skyblue')
        # Add labels and title
        plt.xlabel('Features')
        plt.ylabel('Frequency')
        plt.title('Frequency of features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Save the plot to a file (e.g., PNG format)
        plt.savefig(r".\Plots\random_forest_" + dataset + 'frequencies.png')

        plt.figure(figsize=(8, 6))  # Set the figure size (width, height) in inches
        plt.bar(column_names[:-1], best_freqs_2, color='skyblue')
        # Add labels and title
        plt.xlabel('Features')
        plt.ylabel('Frequency')
        plt.title('Frequency of features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Save the plot to a file (e.g., PNG format)
        plt.savefig(r".\Plots\random_forest_" + dataset + 'frequencies_2.png')