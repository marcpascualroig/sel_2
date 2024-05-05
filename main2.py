import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random_forest import RandomForest
from decision_forest import DecisionForest
import matplotlib.pyplot as plt
import numpy as np


#parameters
test_size=0.3
NT = [1, 10, 25, 50, 75, 100]
random_forest = True
decision_forest= True
trial = False

#datasets = ["ENB2012_data"]
#datasets = ["ContactLens"]
datasets = ["iris", "wdbc", "mushroom"]
#datasets = ["Dry_Bean_Dataset"]
#datasets = ["wdbc"]
#datasets = ["wdbc"]
#datasets=["ionosphere"]



def load_data(dataset):
    if dataset == "mushroom":
        data = pd.read_csv("./Data/" + dataset + ".csv")
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values
        num_instances, tot_features = X.shape
        column_names = data.columns
        column_names = column_names[:-1]
        F1 = [1, 2, int(math.log2(tot_features)) + 1, int(math.sqrt(tot_features))]
        F2 = [int(tot_features / 4), int(tot_features / 2), int(3 * tot_features / 4)]

    if dataset == "wdbc":
        data = pd.read_csv("./Data/" + dataset + ".data")
        X = data.iloc[:, 2:].values  # Features
        y = data.iloc[:, 1].values
        print(X.shape)
        num_instances, tot_features = X.shape
        num_feature_columns = len(data.columns) - 2
        column_names = [i for i in range(num_feature_columns)]
        F1 = [1, 2, int(math.log2(tot_features)) + 1, int(math.sqrt(tot_features))+2]
        F2 = [int(tot_features / 4), int(tot_features / 2), int(3 * tot_features / 4)]

    elif dataset == "iris":
        data = pd.read_csv("./Data/" + dataset + ".data")
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values
        num_instances, tot_features = X.shape
        column_names = ["sepal length", "sepal width", "petal length", "petal width"]
        F1 = [1, 2, 3]
        F2 = [1, 2, 3]

    elif dataset == "ionosphere":
        data = pd.read_csv("./Data/" + dataset + ".data")
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values
        num_instances, tot_features = X.shape
        num_feature_columns = len(data.columns) - 1  # Excluding the target column
        column_names = range(num_feature_columns)
        F1 = [1, 2, int(math.log2(tot_features)) + 1, int(math.sqrt(tot_features))]
        F2 = [int(tot_features / 4), int(tot_features / 2), int(3 * tot_features / 4)]

    elif dataset == "zoo":
        data = pd.read_csv("./Data/" + dataset + ".data")
        X = data.iloc[:, 1:-1].values  # Features
        y = data.iloc[:, -1].values
        column_names = data.columns
        column_names = column_names[1:-1]

    return X, y, column_names, F1, F2



for dataset in datasets:

    print(dataset)
    X, y, column_names, F1, F2 = load_data(dataset)
    print(X)
    print(y)
    print(column_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    num_instances, tot_features = X.shape

    print('n:', num_instances, "M:", tot_features)

    #RANDOM FOREST
    if random_forest:
        accuracy_dic = {}
        freqs_dic = {}
        freqs_2_dic = {}
        plot_accuracy = []
        plot_param = []
        plot_accuracy_2 = []
        plot_param_2 = []

        print("RANDOM FOREST")
        F = sorted(F1)

        for num_trees in NT:
            print("number of trees:", num_trees)
            for num_features in F:
                print("number of features:", num_features)
                forest = RandomForest(tot_features=tot_features, NT=num_trees, F=num_features, max_depth=None)
                freqs, freqs2 = forest.fit(X_train, y_train)
                y_pred = forest.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print("Accuracy:", accuracy)
                #print(freqs)

                accuracy_dic[str(num_trees) + ", " + str(num_features)] = accuracy
                freqs_dic[str(num_trees) + ", " + str(num_features)] = freqs
                freqs_2_dic[str(num_trees) + ", " + str(num_features)] = freqs2

                if num_trees == 50:
                    plot_accuracy.append(accuracy)
                    plot_param.append(num_features)

                if num_features == int(math.log2(tot_features)) + 1:
                    plot_accuracy_2.append(accuracy)
                    plot_param_2.append(num_trees)

        with open(dataset + "_random_forest.txt", 'w') as file:
            file.write("ACCURACY\n")
            for key, value in accuracy_dic.items():
                file.write(f"{key}: {round(value*100, 2)}\n")
            file.write("FREQUENCIES\n")
            for key, value in freqs_dic.items():
                file.write(f"{key} &")
                sorted_features = np.argsort(value)[::-1]
                for index in sorted_features:
                    file.write(str(column_names[index]) + " - ")
                file.write("\\\\ \n")
            file.write("FREQUENCIES 2\n")
            for key, value in freqs_2_dic.items():
                file.write(f"{key} & ")
                sorted_features = np.argsort(value)[::-1]
                for index in sorted_features:
                    file.write(str(column_names[index]) + " - ")
                file.write("\\\\ \n")


        max_key = max(accuracy_dic, key=accuracy_dic.get)
        best_freqs = freqs_dic[max_key]
        best_freqs_2 = freqs_2_dic[max_key]

        # Create a bar plot
        plt.figure(figsize=(8, 6))  # Set the figure size (width, height) in inches
        plt.bar(column_names, best_freqs, color='skyblue')
        # Add labels and title
        plt.xlabel('Features')
        plt.ylabel('Frequency')
        plt.title('Importance of Features (1)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Save the plot to a file (e.g., PNG format)
        plt.savefig(r'.\Plots\random_forest_' + dataset + 'frequencies.png')

        plt.figure(figsize=(8, 6))  # Set the figure size (width, height) in inches
        plt.bar(column_names, best_freqs_2, color='skyblue')
        # Add labels and title
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Importance of Features (2)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Save the plot to a file (e.g., PNG format)
        plt.savefig(r'.\Plots\random_forest_' + dataset + 'frequencies_2.png')


        # Plotting the line plot
        plt.figure(figsize=(8, 6))  # Set the figure size (width, height)
        plt.plot(plot_param, plot_accuracy, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
        # Adding labels and title
        plt.xlabel('Num Features', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy vs. Num Features', fontsize=14)
        # Adding gridlines
        plt.grid(True, linestyle='--', alpha=0.5)
        # Adding annotations for each point
        # Show plot
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off
        plt.savefig(r'.\Plots\random_forest_' + dataset + 'accuracies_1.png')

        # Plotting the line plot
        plt.figure(figsize=(8, 6))  # Set the figure size (width, height)
        plt.plot(plot_param_2, plot_accuracy_2, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
        # Adding labels and title
        plt.xlabel('Num Trees', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy vs. Num Trees', fontsize=14)
        # Adding gridlines
        plt.grid(True, linestyle='--', alpha=0.5)
        # Show plot
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off
        plt.savefig(r'.\Plots\random_forest_' + dataset + 'accuracies_2.png')

    #DECISION FOREST
    if decision_forest:
        accuracy_dic = {}
        freqs_dic = {}
        freqs_2_dic = {}
        plot_accuracy = []
        plot_param = []
        plot_accuracy_2 = []
        plot_param_2 = []
        print("DECISION FOREST")
        F = sorted(F2)
        F.append("random")
        for num_trees in NT:
            print("number of trees:", num_trees)
            for num_features in F:
                print("number of features:", num_features)
                forest = DecisionForest(tot_features=tot_features, NT=num_trees, F=num_features, max_depth=None)
                freqs, freqs2 = forest.fit(X_train, y_train)
                y_pred = forest.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print("Accuracy:", accuracy)
                #print(freqs)

                accuracy_dic[str(num_trees) + ", " + str(num_features)] = accuracy
                freqs_dic[str(num_trees) + ", " + str(num_features)] = freqs
                freqs_2_dic[str(num_trees) + ", " + str(num_features)] = freqs2

                if num_trees == 50:
                    plot_accuracy.append(accuracy)
                    plot_param.append(num_features)

                if num_features == int(3*tot_features/4):
                    plot_accuracy_2.append(accuracy)
                    plot_param_2.append(num_trees)

        with open(dataset + "_decision_forest.txt", 'w') as file:
            file.write("ACCURACY\n")
            for key, value in accuracy_dic.items():
                file.write(f"{key}: {round(value*100, 2)}\n")
            file.write("FREQUENCIES\n")
            for key, value in freqs_dic.items():
                file.write(f"{key} & ")
                sorted_features = np.argsort(value)[::-1]
                for index in sorted_features:
                    file.write(str(column_names[index]) + " - ")
                file.write("\\\\ \n")
            file.write("FREQUENCIES 2\n")
            for key, value in freqs_2_dic.items():
                file.write(f"{key} & ")
                sorted_features = np.argsort(value)[::-1]
                for index in sorted_features:
                    file.write(str(column_names[index]) + " - ")
                file.write("\\\\ \n")



        max_key = max(accuracy_dic, key=accuracy_dic.get)
        best_freqs = freqs_dic[max_key]
        best_freqs_2 = freqs_2_dic[max_key]

        # Create a bar plot
        plt.figure(figsize=(8, 6))  # Set the figure size (width, height) in inches
        plt.bar(column_names, best_freqs, color='skyblue')
        # Add labels and title
        plt.xlabel('Features')
        plt.ylabel('Frequency')
        plt.title('Importance of features (1)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Save the plot to a file (e.g., PNG format)
        plt.savefig(r".\Plots\decision_forest_" + dataset + 'frequencies.png')

        plt.figure(figsize=(8, 6))  # Set the figure size (width, height) in inches
        plt.bar(column_names, best_freqs_2, color='skyblue')
        # Add labels and title
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Importance of features (2)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Save the plot to a file (e.g., PNG format)
        plt.savefig(r".\Plots\decision_forest_" + dataset + 'frequencies_2.png')

        # Plotting the line plot
        plt.figure(figsize=(8, 6))  # Set the figure size (width, height)
        plt.plot(plot_param, plot_accuracy, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
        # Adding labels and title
        plt.xlabel('Num Features', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy vs. Num Features', fontsize=14)
        # Adding gridlines
        plt.grid(True, linestyle='--', alpha=0.5)
        # Adding annotations for each point
        # Show plot
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off
        plt.savefig(r'.\Plots\decision_forest_' + dataset + 'accuracies_1.png')

        # Plotting the line plot
        plt.figure(figsize=(8, 6))  # Set the figure size (width, height)
        plt.plot(plot_param_2, plot_accuracy_2, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
        # Adding labels and title
        plt.xlabel('Num Trees', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy vs. Num Trees', fontsize=14)
        # Adding gridlines
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off
        plt.savefig(r'.\Plots\decision_forest_' + dataset + 'accuracies_2.png')


for dataset in datasets:
    if trial:
        data = pd.read_csv("./Data/" + dataset + ".csv")
        #data = pd.read_excel("./Data/" + dataset + ".xlsx")

        X = data.iloc[:, :-1].values  # Features
        num_instances, tot_features = X.shape
        print('n:', num_instances, "M:", tot_features)

        y = data.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        column_names = data.columns

        num_instances, tot_features = X.shape
        forest = DecisionForest(tot_features=tot_features, NT=1, F=6, max_depth=None)
        freqs, _ = forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        #print(y_pred)
        #print(y_test)
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