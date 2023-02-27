import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Load the Titanic dataset
training_data = pd.read_csv("S:/College Folder/UCF/Spring23/ML/pythonProject/dataset/train.csv")
testing_data = pd.read_csv("S:/College Folder/UCF/Spring23/ML/pythonProject/dataset/test.csv")
# Explore the dataset
print(training_data.head())
print(training_data.describe())
print(training_data.info())

# Preprocess your Titanic training data;

# Treating Missing
print(training_data.isnull().sum())
training_data["Age"].fillna(training_data["Age"].median(), inplace=True)
training_data["Embarked"].fillna(training_data["Embarked"].mode()[0], inplace=True)

# Removing noisy data using Binning

bins = [0, 12, 18, 35, 60, 100]
labels = ["child", "teen", "young adult", "adult", "senior"]
training_data["Age_group"] = pd.cut(training_data["Age"], bins=bins, labels=labels)
age_group_df = training_data.groupby("Age_group").mean()["Survived"].reset_index()

plt.bar(age_group_df["Age_group"], age_group_df["Survived"])
plt.xlabel("Age group")
plt.ylabel("Survival rate")
plt.show()

# Removing outliers by calculating Z-score

z = np.abs(stats.zscore(training_data["Fare"]))
threshold = 3
outliers = training_data[z > threshold]
training_data = training_data[z <= threshold]

plt.hist(training_data["Fare"], bins=50)
plt.title("Fare distribution before removing outliers")
plt.show()
plt.hist(training_data["Fare"], bins=50)
plt.title("Fare distribution after removing outliers")
plt.show()

training_data["Sex"] = training_data["Sex"].map({"male": 0, "female": 1})
training_data["Embarked"] = training_data["Embarked"].map({"S": 0, "C": 1, "Q": 2})
training_data = training_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Age_group"], axis=1)

print("Preprocessed dataset")
print(training_data.head())
print(training_data.describe())

# Learn and fine-tune a decision tree model with the Titanic training data, plot your decision tree;

# Split the dataset into training and testing sets
X = training_data.drop(["Survived"], axis=1)
y = training_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=7, min_samples_split=4, random_state=42)
clf.fit(X_train, y_train)
param_grid = {"max_depth": [3, 5, 7], "min_samples_split": [2, 4, 6]}
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)

plt.figure(figsize=(15, 10))
plot_tree(grid_search.best_estimator_, feature_names=X.columns, filled=True)
plt.show()

y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Apply the five-fold cross validation of your fine-tuned decision tree learning
# model to the Titanic training data to extract average classification accuracy;
scores = cross_val_score(clf, X, y, cv=5)
print("Average classification accuracy(Decision tree):", np.mean(scores))

# Apply the five-fold cross validation of your fine-tuned random forest learning
# model to the Titanic training data to extract average classification accuracy;
clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print("Average classification accuracy(Random Forest):", np.mean(scores))
